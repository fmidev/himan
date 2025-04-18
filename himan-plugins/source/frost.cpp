#include "frost.h"
#include "fetcher.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "numerical_functions.h"
#include "plugin_factory.h"

using namespace std;
using namespace himan;
using namespace himan::plugin;

frost::frost()
{
	itsLogger = logger("frost");
}

void frost::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetParams({param("PROB-FROST-1", aggregation(), processing_type(kProbability)),
	           param("PROB-FROST-2", aggregation(), processing_type(kProbability))});

	Start();
}

shared_ptr<info<double>> BackwardsFetchFromProducer(shared_ptr<plugin_configuration>& cnf, const forecast_type& ftype,
                                                    const forecast_time& ftime, const level& lvl, const param& par,
                                                    int adjust, bool adjustValidTime = false, bool logMissing = true)
{
	auto f = GET_PLUGIN(fetcher);
	shared_ptr<info<double>> ret;
	auto myftime = ftime;
	logger logr("frost");

	for (int i = 0; i < 5; i++)
	{
		try
		{
			ret = f->Fetch(cnf, myftime, lvl, par, ftype, false, true);
			break;
		}
		catch (HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				logr.Trace(fmt::format("Adjusting origin time for {} hours", adjust));
				myftime.OriginDateTime().Adjust(kHourResolution, adjust);
				if (adjustValidTime)
					myftime.ValidDateTime().Adjust(kHourResolution, adjust);
			}
		}
	}

	if (!ret && logMissing)
	{
		logr.Warning(fmt::format("No {} data found for time {}", par.Name(), ftime));
	}

	return ret;
}

forecast_time TruncateToHour(forecast_time ftime, int aStep = 6)
{
	const int latestHour = std::stoi(ftime.OriginDateTime().String("%H"));
	int adjustment = (latestHour - latestHour % aStep) - latestHour;
	ftime.OriginDateTime().Adjust(kHourResolution, adjustment);
	return ftime;
}

void frost::Calculate(shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex)
{
	const param TParam("T-K");
	const param TDParam("TD-K");
	const param TGParam("TG-K");
	param WGParam("FFG-MS", aggregation(HPAggregationType::kMaximum, itsConfiguration->ForecastStep()),
	              processing_type());
	const param T0Param("PROB-TC-0", aggregation(), processing_type(kProbabilityLessThanOrEqual, 273.15));
	const params NParams({param("N-PRCNT"), param("N-0TO1")});
	param RADParam("RADGLO-WM2", aggregation(HPAggregationType::kAverage, itsConfiguration->ForecastStep()),
	               processing_type());
	const param ICNParam("IC-0TO1");
	const param LCParam("LC-0TO1");

	auto myThreadedLogger = logger("frostThread #" + to_string(threadIndex));

	const forecast_time original_forecastTime = myTargetInfo->Time();

	auto ec_forecastTime = TruncateToHour(original_forecastTime, 6);

	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info(
	    fmt::format("Calculating time {}, level {}", original_forecastTime.ValidDateTime(), forecastLevel));

	// Get the latest data from producer defined in configuration file.

	shared_ptr<info<double>> TInfo = Fetch(original_forecastTime, level(kHeight, 2), TParam, forecastType, false);
	if (!TInfo)
	{
		myThreadedLogger.Error("No T-K data found.");
		return;
	}
	shared_ptr<info<double>> TDInfo = Fetch(original_forecastTime, level(kHeight, 2), TDParam, forecastType, false);
	if (!TDInfo)
	{
		myThreadedLogger.Error("No TD-K data found.");
		return;
	}
	shared_ptr<info<double>> NInfo = Fetch(original_forecastTime, level(kHeight, 0), NParams, forecastType, false);
	if (!NInfo)
	{
		myThreadedLogger.Error("No N-PRCNT/N-0TO1 data found.");
		return;
	}

	const double NScale = (NInfo->Param().Name() == "N-PRCNT") ? 1. : 100.;

	// Change forecast origin time for producer 131 due to varying origin time with producer 181.

	auto cnf = make_shared<plugin_configuration>(*itsConfiguration);
	auto f = GET_PLUGIN(fetcher);

	// Get the latest TG-K.

	cnf->SourceProducers({producer(131, 98, 150, "ECG")});

	if (cnf->GetValue("ecmwf_geometry").empty() == false)
	{
		cnf->SourceGeomNames({cnf->GetValue("ecmwf_geometry")});
	}
	else
	{
		cnf->SourceGeomNames({"ECGLO0100", "ECEUR0100"});
	}

	shared_ptr<info<double>> TGInfo =
	    BackwardsFetchFromProducer(cnf, forecastType, ec_forecastTime, level(kGroundDepth, 0, 7), TGParam, -6);

	// Get the latest FFG-MS.

	shared_ptr<info<double>> WGInfo =
	    BackwardsFetchFromProducer(cnf, forecastType, ec_forecastTime, level(kGround, 0), WGParam, -6);

	// Get the latest IC-0TO1.

	shared_ptr<info<double>> ICNInfo =
	    BackwardsFetchFromProducer(cnf, forecastType, ec_forecastTime, level(kGround, 0), ICNParam, -6, false, false);

	// Create ICNInfo when no forecast found.

	if (!ICNInfo)
	{
		ICNInfo = make_shared<info<double>>(forecastType, ec_forecastTime, level(kGround, 0), ICNParam);
		ICNInfo->Producer(myTargetInfo->Producer());
		ICNInfo->Create(myTargetInfo->Base(), true);
		myThreadedLogger.Info("Ice cover information not found, proceeding without it");
	}

	// Get the latest LC-0TO1, available only for hour 00.

	forecast_time LC_time(ec_forecastTime.OriginDateTime(), ec_forecastTime.OriginDateTime());

	shared_ptr<info<double>> LCInfo =
	    BackwardsFetchFromProducer(cnf, forecastType, LC_time, level(kGround, 0), LCParam, -6, true);

	// Get the latest RADGLO-WM2.

	cnf->SourceProducers({producer(240, 86, 240, "ECGMTA")});

	shared_ptr<info<double>> RADInfo =
	    BackwardsFetchFromProducer(cnf, forecastType, ec_forecastTime, level(kHeight, 0), RADParam, -6);

	// Get the latest ECMWF PROB-TC-0. If not found, get earlier.

	forecast_type stat_type = forecast_type(kStatisticalProcessing);

	ec_forecastTime = TruncateToHour(original_forecastTime, 12);

	cnf->SourceProducers({producer(242, 86, 242, "ECM_PROB")});

	if (cnf->GetValue("ecmwfeps_geometry").empty() == false)
	{
		cnf->SourceGeomNames({cnf->GetValue("ecmwfeps_geometry")});
	}
	else
	{
		cnf->SourceGeomNames({"ECGLO0100", "ECEUR0100"});
	}

	shared_ptr<info<double>> T0ECInfo =
	    BackwardsFetchFromProducer(cnf, stat_type, ec_forecastTime, level(kHeight, 2), T0Param, -12, false, false);

	// ECMWF is optional data
	if (!T0ECInfo)
	{
		T0ECInfo = make_shared<info<double>>(stat_type, ec_forecastTime, level(kHeight, 2), T0Param);
		T0ECInfo->Producer(myTargetInfo->Producer());
		T0ECInfo->Create(myTargetInfo->Base(), true);
		myThreadedLogger.Info("ECMWF probabilities not found, proceeding without them");
	}

	// Get the latest MEPS PROB-TC-0 from hour 00, 03, 06, 09, 12, 15, 18 or 21. If not found get earlier.

	shared_ptr<info<double>> T0MEPSInfo = nullptr;

	auto meps_forecastTime = TruncateToHour(original_forecastTime, 3);

	if (meps_forecastTime.Step().Hours() <= 66)
	{
		cnf->SourceProducers({producer(260, 86, 204, "MEPSMTA")});
		if (cnf->GetValue("meps_geometry").empty() == false)
		{
			cnf->SourceGeomNames({cnf->GetValue("meps_geometry")});
		}
		else
		{
			cnf->SourceGeomNames({"MEPS2500D"});
		}

		T0MEPSInfo =
		    BackwardsFetchFromProducer(cnf, stat_type, meps_forecastTime, level(kHeight, 2), T0Param, -3, false, false);
	}

	// MEPS is optional data
	if (!T0MEPSInfo)
	{
		T0MEPSInfo = make_shared<info<double>>(stat_type, meps_forecastTime, level(kHeight, 2), T0Param);
		T0MEPSInfo->Producer(myTargetInfo->Producer());
		T0MEPSInfo->Create(myTargetInfo->Base(), true);
		myThreadedLogger.Info("MEPS probabilities not found, proceeding without them");
	}

	if (!TGInfo || !WGInfo || !ICNInfo || !LCInfo || !RADInfo || !T0ECInfo || !T0MEPSInfo)
	{
		myThreadedLogger.Error(fmt::format("Skipping step {}, level {}", original_forecastTime.Step(), forecastLevel));
		return;
	}

	const int month = stoi(original_forecastTime.ValidDateTime().String("%m"));
	const int day = stoi(original_forecastTime.ValidDateTime().String("%d"));

	myThreadedLogger.Info("Got all needed data");

	using numerical_functions::RampDown;

	LOCKSTEP(myTargetInfo, TInfo, TDInfo, TGInfo, WGInfo, NInfo, ICNInfo, LCInfo, RADInfo, T0ECInfo, T0MEPSInfo)

	{
		double T = TInfo->Value() - himan::constants::kKelvin;
		double TD = TDInfo->Value() - himan::constants::kKelvin;
		double TG = TGInfo->Value() - himan::constants::kKelvin;
		double WG = WGInfo->Value();
		double N = NInfo->Value() * NScale;
		double ICN = ICNInfo->Value();
		double LC = LCInfo->Value();
		double RAD = RADInfo->Value();
		double T0EC = T0ECInfo->Value();
		double T0MEPS = T0MEPSInfo->Value();

		if (IsMissingValue({T, TD, TG, WG, N, LC, RAD}))
		{
			continue;
		}

		// Calculating indexes and coefficients.

		// dewIndex

		const double dewIndex = RampDown(-5.0, 5.0, TD);  // TD -5...5

		// nIndex

		const double nIndex = (100.0 - N) / 100.0;

		// tIndexHigh

		const double tIndexHigh = RampDown(2.5, 15.0, T) * RampDown(2.5, 15.0, T);

		// wgWind

		const double wgWind = RampDown(1.0, 6.0, WG);

		double lowWindCoef = 1.0;
		double nCoef = 1.0;
		double weight = 4.0;
		double stabCoef = 1.5 * wgWind + 1.0;

		// Adjusting coefficients when T above and near zero.

		if (T >= 0 && T < 2.5)
		{
			lowWindCoef = RampDown(0., 2.5, T) * weight + 1.0;
			nCoef = 1.0 / lowWindCoef;
		}

		// Calculating frost and severe frost probability.

		double frost_prob;

		if (T < -3.0)
		{
			frost_prob = 1.0;
		}

		else if (T < 0)
		{
			frost_prob = 0.9;
		}

		else
		{
			frost_prob = ((lowWindCoef * dewIndex) + stabCoef + (nCoef * nIndex)) /
			             (lowWindCoef + stabCoef + (1.0 / lowWindCoef)) * tIndexHigh;
		}

		// Raising the frost probability due to ground temperature TG.

		double tgModel = MissingDouble();

		if (T < 5.0)
		{
			tgModel = sqrt(RampDown(-6.0, 5.0, TG));
		}

		if (frost_prob < tgModel)
		{
			frost_prob = tgModel;
		}

		// Raising the frost probability due to probability of T<0 and T. Both EC and MEPS cases.

		if (T0EC > 0.6 && T < 5.0 && frost_prob < T0EC)
		{
			frost_prob = T0EC;
		}

		if (frost_prob < T0MEPS && T0MEPS > 0.4)
		{
			frost_prob = (frost_prob * 2.0 + T0MEPS) / 3.0;
		}

		double severe_frost_prob = pow(frost_prob * 100, 3) * 1e-4 * 0.01;

		// No frost when radiation is high enough. Valid in  1.4.-15.9.

		if ((month >= 4 && month <= 8) || (month == 9 && day <= 15))
		{
			if (RAD > 175)
			{
				frost_prob = 0.0;
				severe_frost_prob = 0.0;
			}
		}

		// Lowering frost probability due to sun's elevation angle. Valid in 1.4.-15.9.

		/*if ((month >= 4 && month <= 8) || (month == 9 && day <= 15))
		{
		    NFmiLocation theLocation(myTargetInfo->LatLon().X(), myTargetInfo->LatLon().Y());

		    double elevationAngle = theLocation.ElevationAngle(theTime);
		    double angleCoef = kHPMissingValue;

		    angleCoef = RampDown(-1.0, 20.0, elevationAngle);

		    frost_prob = angleCoef * frost_prob;
		}*/

		// No frost probability on sea when there is no ice.

		if (ICN == 0 && LC == 0)
		{
			frost_prob = 0;
			severe_frost_prob = 0;
		}

		// Lowering frost probability when forecasted T is high enough.

		if (T > 6.0)
		{
			frost_prob = frost_prob * RampDown(6.0, 15.0, T);
		}

		severe_frost_prob = min(frost_prob, severe_frost_prob);

		// Simple quality control

		if (frost_prob < 0.0 || frost_prob > 1.0)
		{
			frost_prob = MissingDouble();
		}

		if (severe_frost_prob < 0.0 || severe_frost_prob > 1.0)
		{
			severe_frost_prob = MissingDouble();
		}

		myTargetInfo->Index<param>(0);
		myTargetInfo->Value(frost_prob);
		myTargetInfo->Index<param>(1);
		myTargetInfo->Value(severe_frost_prob);
	}

	myThreadedLogger.Info(
	    fmt::format("[CPU] Missing values: {}/{}", myTargetInfo->Data().MissingCount(), myTargetInfo->Data().Size()));
}
