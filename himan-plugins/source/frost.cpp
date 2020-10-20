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

void frost::Calculate(shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex)
{
	/*NFmiMetTime theTime(static_cast<short>(stoi(myTargetInfo->Time().ValidDateTime().String("%Y"))),
	                    static_cast<short>(stoi(myTargetInfo->Time().ValidDateTime().String("%m"))),
	                    static_cast<short>(stoi(myTargetInfo->Time().ValidDateTime().String("%d"))),
	                    static_cast<short>(stoi(myTargetInfo->Time().ValidDateTime().String("%H"))),
	                    static_cast<short>(stoi(myTargetInfo->Time().ValidDateTime().String("%M"))));*/

	const param TParam("T-K");
	const param TDParam("TD-K");
	const param TGParam("TG-K");
	const param WGParam("FFG-MS");
	const param T0Param("PROB-TC-0");
	const param NParam("N-PRCNT");
	const param RADParam("RADGLO-WM2");
	const param ICNParam("IC-0TO1");
	const param LCParam("LC-0TO1");

	auto myThreadedLogger = logger("frostThread #" + to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	forecast_time original_forecastTime(forecastTime);
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + ", level " +
	                      static_cast<string>(forecastLevel));

	// Get the latest data from producer 181.

	info_t TInfo = Fetch(forecastTime, level(kHeight, 2), TParam, forecastType, false);
	if (!TInfo)
	{
		myThreadedLogger.Error("No T-K data found.");
		return;
	}
	info_t TDInfo = Fetch(forecastTime, level(kHeight, 2), TDParam, forecastType, false);
	if (!TDInfo)
	{
		myThreadedLogger.Error("No TD-K data found.");
		return;
	}
	info_t NInfo = Fetch(forecastTime, level(kHeight, 0), NParam, forecastType, false);
	if (!NInfo)
	{
		myThreadedLogger.Error("No N-PRCNT data found.");
		return;
	}

	// Change forecast origin time for producer 131 due to varying origin time with producer 181.

	auto cnf = make_shared<plugin_configuration>(*itsConfiguration);
	auto f = GET_PLUGIN(fetcher);

	// Get the latest TG-K.

	info_t TGInfo;
	int i = 0;
	bool success = false;

	while (success == false)
	{
		try
		{
			cnf->SourceProducers({producer(131, 98, 150, "ECG")});
			TGInfo = f->Fetch(cnf, forecastTime, level(kGroundDepth, 0, 7), TGParam, forecastType, false);
			success = true;
		}
		catch (HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				myThreadedLogger.Error("No TG-K data found.");
				forecastTime.OriginDateTime().Adjust(kHourResolution, -1);
				i++;
			}
			if (i > 100)
			{
				return;
			}
		}
	}
	// Get the latest FFG-MS.

	info_t WGInfo;

	try
	{
		cnf->SourceProducers({producer(131, 98, 150, "ECG")});
		WGInfo = f->Fetch(cnf, forecastTime, level(kGround, 0), WGParam, forecastType, false);
	}
	catch (HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
			myThreadedLogger.Error("No FFG-MS data found.");
		}
		return;
	}

	// Get the latest IC-0TO1.

	info_t ICNInfo;

	bool ICN_forecast = true;

	// Change forecast origin time to 00 or 12 if necessary.

	int latestHour = std::stoi(forecastTime.OriginDateTime().String("%H"));

	if (latestHour > 0 && latestHour < 12)
	{
		forecastTime.OriginDateTime().Adjust(kHourResolution, -latestHour);
	}
	if (latestHour > 12 && latestHour <= 23)
	{
		forecastTime.OriginDateTime().Adjust(kHourResolution, -latestHour);
		forecastTime.OriginDateTime().Adjust(kHourResolution, 12);
	}

	try
	{
		cnf->SourceProducers({producer(131, 98, 150, "ECG")});
		ICNInfo = f->Fetch(cnf, forecastTime, level(kGround, 0), ICNParam, forecastType, false);
	}
	catch (HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
			myThreadedLogger.Error("No IC-0TO1 data found.");
			ICN_forecast = false;
		}
	}

	// Create ICNInfo when no forecast found.

	if (!ICNInfo)
	{
		ICNInfo = make_shared<info<double>>(forecastType, forecastTime, level(kGround, 0), ICNParam);
		ICNInfo->Producer(myTargetInfo->Producer());
		ICNInfo->Create(myTargetInfo->Base(), true);
	}

	// Get the latest LC-0TO1, available only for hour 00.

	info_t LCInfo;

	forecast_time LC_time(forecastTime.OriginDateTime(), forecastTime.OriginDateTime());

	try
	{
		cnf->SourceProducers({producer(131, 98, 150, "ECG")});
		cnf->SourceGeomNames({"ECEUR0100"});
		LCInfo = f->Fetch(cnf, LC_time, level(kGround, 0), LCParam, forecastType, false);
	}
	catch (HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
			myThreadedLogger.Error("No LC-0TO1 data found.");
		}
		return;
	}

	// Get the latest RADGLO-WM2.

	info_t RADInfo;

	forecastTime = original_forecastTime;

	success = false;
	i = 0;

	while (success == false)
	{
		try
		{
			cnf->SourceProducers({producer(240, 86, 240, "ECGMTA")});
			cnf->SourceGeomNames({"ECGLO0100"});
			RADInfo = f->Fetch(cnf, forecastTime, level(kHeight, 0), RADParam, forecastType, false);
			success = true;
		}
		catch (HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				myThreadedLogger.Error("No RADGLO-WM2 data found.");
				forecastTime.OriginDateTime().Adjust(kHourResolution, -1);
				i++;
			}
			if (i > 100)
			{
				return;
			}
		}
	}

	// Get the latest ECMWF PROB-TC-0. If not found, get earlier.

	info_t T0ECInfo;

	forecast_type stat_type = forecast_type(kStatisticalProcessing);
	forecastTime = original_forecastTime;

	i = 0;
	success = false;

	while (success == false)
	{
		try
		{
			// ECMWF PROB-TC-0 is calculated only for every 3 hours.

			int forecastHour = std::stoi(forecastTime.ValidDateTime().String("%H"));

			if (forecastHour % 3 == 1 || forecastHour % 3 == 2)
			{
				myThreadedLogger.Error("ECMWF PROB-TC-0 not available for forecast hour: " +
				                       forecastTime.ValidDateTime().String("%H"));
				return;
			}

			cnf->SourceProducers({producer(242, 86, 242, "ECM_PROB")});
			cnf->SourceGeomNames({"ECEUR0200"});
			T0ECInfo = f->Fetch(cnf, forecastTime, level(kGround, 0), T0Param, stat_type, false);
			success = true;
		}
		catch (HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				const string analtime = forecastTime.OriginDateTime().String("%Y-%m-%d %H:%M:%S");
				myThreadedLogger.Error("ECMWF PROB-TC-0 from analysis time " + analtime + " not found.");
				forecastTime.OriginDateTime().Adjust(kHourResolution, -1);
				i++;
			}
			if (i > 100)
			{
				return;
			}
		}
		catch (...)
		{
			return;
		}
	}

	// Get the latest MEPS PROB-TC-0 from hour 00, 03, 06, 09, 12, 15, 18 or 21. If not found get earlier.

	info_t T0MEPSInfo;

	forecastTime = original_forecastTime;
	i = 0;
	success = false;
	bool MEPS_forecast = true;

	while (success == false)
	{
		try
		{
			latestHour = std::stoi(forecastTime.OriginDateTime().String("%H"));

			if (latestHour == 1 || latestHour == 4 || latestHour == 7 || latestHour == 10 || latestHour == 13 ||
			    latestHour == 16 || latestHour == 19 || latestHour == 22)
			{
				forecastTime.OriginDateTime().Adjust(kHourResolution, -1);
			}

			if (latestHour == 2 || latestHour == 5 || latestHour == 8 || latestHour == 11 || latestHour == 14 ||
			    latestHour == 17 || latestHour == 20 || latestHour == 23)
			{
				forecastTime.OriginDateTime().Adjust(kHourResolution, -2);
			}

			cnf->SourceProducers({producer(260, 86, 204, "MEPSMTA")});
			cnf->SourceGeomNames({"MEPS2500D"});
			T0MEPSInfo = f->Fetch(cnf, forecastTime, level(kHeight, 2), T0Param, stat_type, false);
			success = true;
		}
		catch (HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				const string analtime = forecastTime.OriginDateTime().String("%Y-%m-%d %H:%M:%S");
				myThreadedLogger.Error("MEPS PROB-TC-0 from analysis time " + analtime + " not found.");
				if (i == 0)
				{
					forecastTime.OriginDateTime().Adjust(kHourResolution, 24);
					i++;
				}
				else
				{
					forecastTime.OriginDateTime().Adjust(kHourResolution, -1);
					i++;
				}
			}
			if (i > 30)
			{
				MEPS_forecast = false;
				break;
			}
		}
		catch (...)
		{
			return;
		}
	}

	// Create T0MEPSInfo when no forecast found.

	if (!T0MEPSInfo)
	{
		T0MEPSInfo = make_shared<info<double>>(stat_type, original_forecastTime, level(kHeight, 2), T0Param);
		T0MEPSInfo->Producer(myTargetInfo->Producer());
		T0MEPSInfo->Create(myTargetInfo->Base(), true);
	}

	if (!TGInfo || !WGInfo || !ICNInfo || !LCInfo || !RADInfo || !T0ECInfo || !T0MEPSInfo)
	{
		myThreadedLogger.Warning("Skipping step " + static_cast<string>(forecastTime.Step()) + ", level " +
		                         static_cast<string>(forecastLevel));
		return;
	}

	string deviceType = "CPU";

	LOCKSTEP(myTargetInfo, TInfo, TDInfo, TGInfo, WGInfo, NInfo, ICNInfo, LCInfo, RADInfo, T0ECInfo, T0MEPSInfo)

	{
		double T = TInfo->Value() - himan::constants::kKelvin;
		double TD = TDInfo->Value() - himan::constants::kKelvin;
		double TG = TGInfo->Value() - himan::constants::kKelvin;
		double WG = WGInfo->Value();
		double N = NInfo->Value();
		double ICN;
		if (!ICN_forecast)  // No forecast available.
		{
			ICN = kHPMissingValue;
		}
		else
		{
			ICN = ICNInfo->Value();
		}
		double LC = LCInfo->Value();
		double RAD = RADInfo->Value();
		double T0EC = T0ECInfo->Value();
		double T0MEPS;
		if (!MEPS_forecast)  // No forecast available.
		{
			T0MEPS = kHPMissingValue;
		}
		else
		{
			T0MEPS = T0MEPSInfo->Value();
			if (IsMissingValue({T0MEPS}))  // MEPS hour 66 case.
			{
				T0MEPS = kHPMissingValue;
			}
		}

		if (IsMissingValue({T, TD, TG, WG, N, ICN, LC, RAD, T0EC, T0MEPS}))
		{
			continue;
		}

		// Calculating indexes and coefficients.

		// dewIndex

		double dewIndex = kHPMissingValue;

		dewIndex = rampDown(-5.0, 5.0, TD);  // TD -5...5

		// nIndex

		double nIndex = kHPMissingValue;

		nIndex = (100.0 - N) / 100.0;

		// tIndexHigh

		double tIndexHigh = kHPMissingValue;

		tIndexHigh = rampDown(2.5, 15.0, T) * rampDown(2.5, 15.0, T);

		// wgWind

		double wgWind = kHPMissingValue;

		wgWind = rampDown(1.0, 6.0, WG);

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

		double tgModel = kHPMissingValue;

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

		int month = stoi(forecastTime.ValidDateTime().String("%m"));
		int day = stoi(forecastTime.ValidDateTime().String("%d"));

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
