/**
 * @file ncl.cpp
 *
 */

#include "ncl.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "plugin_factory.h"
#include <boost/lexical_cast.hpp>

#include "neons.h"
#include "radon.h"

using namespace std;
using namespace himan::plugin;

const string itsName("ncl");

ncl::ncl() : itsBottomLevel(kHPMissingInt), itsTopLevel(kHPMissingInt), itsTargetTemperature(kHPMissingInt)
{
	itsLogger = logger(itsName);
}

void ncl::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	HPDatabaseType dbtype = conf->DatabaseType();

	if (dbtype == kNeons || dbtype == kNeonsAndRadon)
	{
		auto n = GET_PLUGIN(neons);

		itsBottomLevel = boost::lexical_cast<int>(
		    n->ProducerMetaData(itsConfiguration->SourceProducer().Id(), "last hybrid level number"));
		itsTopLevel = boost::lexical_cast<int>(
		    n->ProducerMetaData(itsConfiguration->SourceProducer().Id(), "first hybrid level number"));
	}

	if ((dbtype == kRadon || dbtype == kNeonsAndRadon) &&
	    (itsBottomLevel == kHPMissingInt || itsTopLevel == kHPMissingInt))
	{
		auto r = GET_PLUGIN(radon);

		itsBottomLevel = boost::lexical_cast<int>(
		    r->RadonDB().GetProducerMetaData(itsConfiguration->SourceProducer().Id(), "last hybrid level number"));
		itsTopLevel = boost::lexical_cast<int>(
		    r->RadonDB().GetProducerMetaData(itsConfiguration->SourceProducer().Id(), "first hybrid level number"));
	}

	param theRequestedParam;

	if (itsConfiguration->Exists("temp") && itsConfiguration->GetValue("temp") == "-20")
	{
		theRequestedParam.Name("HM20C-M");
		theRequestedParam.UnivId(28);
		itsTargetTemperature = -20;
	}

	if (itsConfiguration->Exists("temp") && itsConfiguration->GetValue("temp") == "0")
	{
		theRequestedParam.Name("H0C-M");
		theRequestedParam.UnivId(270);
		itsTargetTemperature = 0;
	}

	SetParams({theRequestedParam});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void ncl::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	const param HParam("HL-M");
	const param TParam("T-K");

	int levelNumber = itsBottomLevel;

	level HLevel(himan::kHybrid, static_cast<float>(levelNumber), "HYBRID");

	auto myThreadedLogger = logger(itsName + "Thread #" + to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
						  static_cast<string>(forecastLevel));

	info_t HInfo = Fetch(forecastTime, HLevel, HParam, forecastType, false);
	info_t TInfo = Fetch(forecastTime, HLevel, TParam, forecastType, false);

	if (!HInfo || !TInfo)
	{
		myThreadedLogger.Error("Skipping step " + to_string(forecastTime.Step()) + ", level " +
							   static_cast<string>(forecastLevel));
		return;
	}

	bool firstLevel = true;

	myTargetInfo->Data().Fill(-1);

	HInfo->ResetLocation();
	TInfo->ResetLocation();

	level curLevel = HLevel;
	level prevLevel;

	string deviceType = "CPU";

	info_t prevHInfo, prevTInfo;

	while (--levelNumber >= itsTopLevel)
	{
		myThreadedLogger.Trace("Level: " + to_string(levelNumber));

		if (prevHInfo)
		{
			prevHInfo->ResetLocation();
		}

		if (prevTInfo)
		{
			prevTInfo->ResetLocation();
		}

		assert(HInfo && TInfo);

		LOCKSTEP(myTargetInfo, TInfo, HInfo)
		{
			double height = HInfo->Value();
			double temp = TInfo->Value();

			double prevHeight(MissingDouble());
			double prevTemp(MissingDouble());

			double targetHeight = myTargetInfo->Value();

			if (!firstLevel)
			{
				prevHInfo->NextLocation();
				prevHeight = prevHInfo->Value();

				prevTInfo->NextLocation();
				prevTemp = prevTInfo->Value();
			}

			if (IsMissingValue({height, temp}) || (!firstLevel && (IsMissingValue({prevHeight, prevTemp}))))
			{
				continue;
			}

			temp -= himan::constants::kKelvin;
			prevTemp -= himan::constants::kKelvin;

			if (targetHeight != -1 || IsMissing(targetHeight))
			{
				if (temp >= itsTargetTemperature)  // && levelNumber >= (itsBottomLevel - 5))
				{
					/*
					 * Lowest 5 model levels and "inversion fix".
					 *
					 * Especially in winter time surface inversion is common: temperature
					 * close to ground surfaceis colder than the temperature above it.
					 * This inversion is usually very limited in height, maybe 10 .. 200
					 * meters. When calculating height of zero level, we can take into
					 * account this inversion so we test if the temperature at lowest level
					 * is already below target temperature (0 or -20), and if that is the
					 * situation we do not directly set height to MissingDouble() but
					 * move onwards and only if the temperature stays below target for
					 * the first 5 levels (in Hirlam lowest hybrid levels are spaced ~30m
					 * apart) we can say that surface inversion does not exist and the
					 * air temperature is simply too cold for this target temperature.
					 *
					 * In order to imitate hilake however the inversion fix limitation for
					 * bottom 5 has been disabled, since what hilake does is it applies
					 * the fix as long as the process is running, at least up to hybrid
					 * level 29 which on Hirlam is on average 3700m above ground.
					 *
					 * This seems like a non-optimal approach but this is how hilake
					 * does it and until we have confirmation that the inversion
					 * height should be limited, use the hilake way also in himan.
					 *
					 * Inversion: http://blogi.foreca.fi/2012/01/pakkanen-ja-inversio/
					 */

					myTargetInfo->Value(-1);
				}

				// No inversion fix, value already found for this gridpoint
				// (real value or missing value)

				continue;
			}

			if ((firstLevel && temp < itsTargetTemperature) ||
			    (prevTemp < itsTargetTemperature && temp < itsTargetTemperature))
			{
				/*
				 * Height is below ground (first model level) or its so cold
				 * in the lower atmosphere that we cannot find the requested
				 * temperature, set height to to missing value.
				 */

				targetHeight = MissingDouble();
			}
			else if (prevTemp > itsTargetTemperature && temp < itsTargetTemperature)
			{
				// Found value, interpolate

				double p_rel = (itsTargetTemperature - temp) / (prevTemp - temp);
				targetHeight = height + (prevHeight - height) * p_rel;
			}
			else
			{
				// Too warm
				continue;
			}

			myTargetInfo->Value(targetHeight);
		}

		if (CountValues(myTargetInfo))
		{
			break;
		}

		prevLevel = curLevel;
		curLevel = level(himan::kHybrid, static_cast<float>(levelNumber), "HYBRID");

		HInfo = Fetch(forecastTime, curLevel, HParam, forecastType, false);
		TInfo = Fetch(forecastTime, curLevel, TParam, forecastType, false);

		prevHInfo = Fetch(forecastTime, prevLevel, HParam, forecastType, false);
		prevTInfo = Fetch(forecastTime, prevLevel, TParam, forecastType, false);

		if (!HInfo || !TInfo || !prevHInfo || !prevTInfo)
		{
			myThreadedLogger.Error("Not enough data for step " + to_string(forecastTime.Step()) +
								   ", level " + static_cast<string>(forecastLevel));
			break;
		}

		firstLevel = false;
	}

	/*
	 * Replaces all unset values
	 */

	myTargetInfo->ResetLocation();

	while (myTargetInfo->NextLocation())
	{
		if (myTargetInfo->Value() == -1.)
		{
			myTargetInfo->Value(MissingDouble());
		}
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}

bool ncl::CountValues(const shared_ptr<himan::info> values)
{
	size_t s = values->Data().Size();

#ifdef DEBUG
	size_t foundVals = s;
#endif

	for (size_t j = 0; j < s; j++)
	{
		if (values->Data().At(j) == -1)
		{
#ifdef DEBUG
			foundVals--;
#else
			return false;
#endif
		}
	}

#ifdef DEBUG
	itsLogger.Debug("Found value for " + to_string(foundVals) + "/" + to_string(s) + " gridpoints");

	if (foundVals != s) return false;
#endif

	return true;
}
