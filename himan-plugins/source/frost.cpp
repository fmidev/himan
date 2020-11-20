#include "frost.h"
#include "fetcher.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "plugin_factory.h"
//#include <NFmiLocation.h>
//#include <NFmiMetTime.h>

using namespace std;
using namespace himan;
using namespace himan::plugin;

// rampDown function returns a value between 1 and 0,
// depending where valueInBetween is in the interval between start and end.

double rampDown(const double& start, const double& end, const double& valueInBetween)
{
	if (valueInBetween <= start)
		return 1.0;

	if (valueInBetween >= end)
		return 0.0;

	return (end - valueInBetween) / (end - start);
}

frost::frost()
{
	itsLogger = logger("frost");
}

void frost::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetParams({param("PROB-FROST-1"), param("PROB-FROST-2")});

	Start();
}

info_t BackwardsFetchFromProducer(shared_ptr<plugin_configuration>& cnf, const forecast_type& ftype,
                                  const forecast_time& ftime, const level& lvl, const param& par, int adjust,
                                  bool adjustValidTime = false)
{
	auto f = GET_PLUGIN(fetcher);
	info_t ret;
	auto myftime = ftime;
	logger logr("frost");

	for (int i = 0; i < 6; i++)
	{
		try
		{
			ret = f->Fetch(cnf, myftime, lvl, par, ftype, false);
			break;
		}
		catch (HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				logr.Debug(fmt::format("Adjusting origin time for {} hours", adjust));
				myftime.OriginDateTime().Adjust(kHourResolution, adjust);
				if (adjustValidTime)
					myftime.ValidDateTime().Adjust(kHourResolution, adjust);
			}
		}
	}

	if (!ret)
	{
		logr.Warning(fmt::format("No {} data found", par.Name()));
	}

	return ret;
}

void frost::Calculate(shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex)
{
	const param TParam("T-K");
	const param TDParam("TD-K");
	const param TGParam("TG-K");
	const param WGParam("FFG-MS");
	const param T0Param("PROB-TC-0");
	const params NParams({param("N-PRCNT"), param("N-0TO1")});
	const param RADParam("RADGLO-WM2");
	const param ICNParam("IC-0TO1");
	const param LCParam("LC-0TO1");

	auto myThreadedLogger = logger("frostThread #" + to_string(threadIndex));

	const forecast_time original_forecastTime = myTargetInfo->Time();
	auto ec_forecastTime = original_forecastTime;
	const int latestHour = std::stoi(ec_forecastTime.OriginDateTime().String("%H"));
	int adjustment = (latestHour - latestHour % 6) - latestHour;
	ec_forecastTime.OriginDateTime().Adjust(kHourResolution, adjustment);

	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info(fmt::format("Calculating time {}, level {}",
	                                  static_cast<string>(original_forecastTime.ValidDateTime()),
	                                  static_cast<string>(forecastLevel)));

	// Get the latest data from producer defined in configuration file.

	info_t TInfo = Fetch(original_forecastTime, level(kHeight, 2), TParam, forecastType, false);
	if (!TInfo)
	{
		myThreadedLogger.Error("No T-K data found.");
		return;
	}
	info_t TDInfo = Fetch(original_forecastTime, level(kHeight, 2), TDParam, forecastType, false);
	if (!TDInfo)
	{
		myThreadedLogger.Error("No TD-K data found.");
		return;
	}
	info_t NInfo = Fetch(original_forecastTime, level(kHeight, 0), NParams, forecastType, false);
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
	cnf->SourceGeomNames({"ECGLO0100", "ECEUR0100"});

	info_t TGInfo =
	    BackwardsFetchFromProducer(cnf, forecastType, ec_forecastTime, level(kGroundDepth, 0, 7), TGParam, -6);

	// Get the latest FFG-MS.

	info_t WGInfo = BackwardsFetchFromProducer(cnf, forecastType, ec_forecastTime, level(kGround, 0), WGParam, -6);

	// Get the latest IC-0TO1.

	info_t ICNInfo = BackwardsFetchFromProducer(cnf, forecastType, ec_forecastTime, level(kGround, 0), ICNParam, -6);

	// Create ICNInfo when no forecast found.

	if (!ICNInfo)
	{
		ICNInfo = make_shared<info<double>>(forecastType, ec_forecastTime, level(kGround, 0), ICNParam);
		ICNInfo->Producer(myTargetInfo->Producer());
		ICNInfo->Create(myTargetInfo->Base(), true);
	}

	// Get the latest LC-0TO1, available only for hour 00.

	forecast_time LC_time(ec_forecastTime.OriginDateTime(), ec_forecastTime.OriginDateTime());

	info_t LCInfo = BackwardsFetchFromProducer(cnf, forecastType, LC_time, level(kGround, 0), LCParam, -6, true);

	// Get the latest RADGLO-WM2.

	cnf->SourceProducers({producer(240, 86, 240, "ECGMTA")});

	info_t RADInfo = BackwardsFetchFromProducer(cnf, forecastType, ec_forecastTime, level(kHeight, 0), RADParam, -6);

	// Get the latest ECMWF PROB-TC-0. If not found, get earlier.

	forecast_type stat_type = forecast_type(kStatisticalProcessing);

	// ECMWF PROB-TC-0 is calculated only for every 3 hours.

	int forecastHour = std::stoi(ec_forecastTime.ValidDateTime().String("%H"));

	if (forecastHour % 3 == 1 || forecastHour % 3 == 2)
	{
		myThreadedLogger.Error(fmt::format("ECMWF PROB-TC-0 not available for forecast hour: {}",
		                                   ec_forecastTime.ValidDateTime().String("%H")));
		return;
	}
	ec_forecastTime = original_forecastTime;
	adjustment = (latestHour - latestHour % 12) - latestHour;
	ec_forecastTime.OriginDateTime().Adjust(kHourResolution, adjustment);

	cnf->SourceProducers({producer(242, 86, 242, "ECM_PROB")});
	cnf->SourceGeomNames({"ECGLO0200", "ECEUR0200"});
	info_t T0ECInfo = BackwardsFetchFromProducer(cnf, stat_type, ec_forecastTime, level(kGround, 0), T0Param, -12);

	// Get the latest MEPS PROB-TC-0 from hour 00, 03, 06, 09, 12, 15, 18 or 21. If not found get earlier.

	auto meps_forecastTime = original_forecastTime;
	adjustment = (latestHour - latestHour % 3) - latestHour;
	meps_forecastTime.OriginDateTime().Adjust(kHourResolution, adjustment);

	cnf->SourceProducers({producer(260, 86, 204, "MEPSMTA")});
	cnf->SourceGeomNames({"MEPS2500D"});

	info_t T0MEPSInfo = BackwardsFetchFromProducer(cnf, stat_type, meps_forecastTime, level(kHeight, 2), T0Param, -3);

	// MEPS is optional data
	if (!T0MEPSInfo)
	{
		T0MEPSInfo = make_shared<info<double>>(stat_type, meps_forecastTime, level(kHeight, 2), T0Param);
		T0MEPSInfo->Producer(myTargetInfo->Producer());
		T0MEPSInfo->Create(myTargetInfo->Base(), true);
	}

	if (!TGInfo || !WGInfo || !ICNInfo || !LCInfo || !RADInfo || !T0ECInfo || !T0MEPSInfo)
	{
		myThreadedLogger.Warning(fmt::format("Skipping step {}, level {}",
		                                     static_cast<string>(original_forecastTime.Step()),
		                                     static_cast<string>(forecastLevel)));
		return;
	}

	string deviceType = "CPU";

	const int month = stoi(original_forecastTime.ValidDateTime().String("%m"));
	const int day = stoi(original_forecastTime.ValidDateTime().String("%d"));

	myThreadedLogger.Info("Got all needed data");

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

		if (IsMissingValue({T, TD, TG, WG, N, LC, RAD, T0EC}))
		{
			continue;
		}

		// Calculating indexes and coefficients.

		// dewIndex

		const double dewIndex = rampDown(-5.0, 5.0, TD);  // TD -5...5

		// nIndex

		const double nIndex = (100.0 - N) / 100.0;

		// tIndexHigh

		const double tIndexHigh = rampDown(2.5, 15.0, T) * rampDown(2.5, 15.0, T);

		// wgWind

		const double wgWind = rampDown(1.0, 6.0, WG);

		double lowWindCoef = 1.0;
		double nCoef = 1.0;
		double weight = 4.0;
		double stabCoef = 1.5 * wgWind + 1.0;

		// Adjusting coefficients when T above and near zero.

		if (T >= 0 && T < 2.5)
		{
			lowWindCoef = rampDown(0, 2.5, T) * weight + 1.0;
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
			tgModel = sqrt(rampDown(-6.0, 5.0, TG));
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

		    angleCoef = rampDown(-1.0, 20.0, elevationAngle);

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
			frost_prob = frost_prob * rampDown(6.0, 15.0, T);
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

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
