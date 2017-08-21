/**
 * @file weather_code_1
 *
 */

#include "weather_code_1.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "metutil.h"

using namespace std;
using namespace himan::plugin;

const string itsName("weather_code_1");

// Required source parameters

const himan::param ZParam("Z-M2S2");
const himan::params NParams({himan::param("N-0TO1"), himan::param("N-PRCNT")});
const himan::param TParam("T-K");
const himan::param CloudParam("CLDSYM-N");
const himan::param KindexParam("KINDEX-N");

// ..and their levels
const himan::level Z1000Level(himan::kPressure, 1000, "PRESSURE");
const himan::level Z850Level(himan::kPressure, 850, "PRESSURE");
const himan::level T2Level(himan::kHeight, 2, "HEIGHT");
const himan::level NLevel(himan::kHeight, 0, "HEIGHT");

weather_code_1::weather_code_1()
{
	itsLogger = logger(itsName);
}

void weather_code_1::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetParams({param("HSADE1-N", 52)});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void weather_code_1::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	auto myThreadedLogger = logger(itsName + "Thread #" + to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                      static_cast<string>(forecastLevel));

	/*
	 * In order to know which source precipitation parameter should be used we need
	 * to know the forecast step. This of course applies only to those forecasts that
	 * have a varying time step.
	 *
	 * First try to get step from configuration file key 'step'. If that is not available,
	 * try to determine the step by comparing earlier or later times with the current time.
	 * If this doesn't work then default to one hour time step.
	 */

	int paramStep = itsConfiguration->ForecastStep();

	if (paramStep == kHPMissingInt)
	{
		// shit

		if (myTargetInfo->SizeTimes() == 1)
		{
			paramStep = 1;
			itsLogger.Warning("Unable to determine step from current configuration, assuming one hour");
		}
		else
		{
			if (myTargetInfo->TimeIndex() == 0)
			{
				forecast_time otherTime = myTargetInfo->PeekTime(myTargetInfo->TimeIndex() + 1);
				paramStep = otherTime.Step() - myTargetInfo->Time().Step();
			}
			else
			{
				forecast_time otherTime = myTargetInfo->PeekTime(myTargetInfo->TimeIndex() - 1);
				paramStep = myTargetInfo->Time().Step() - otherTime.Step();
			}
		}
	}

	param RRParam("RR-1-MM");  // Default

	if (paramStep == 3)
	{
		RRParam = param("RR-3-MM");
	}
	else if (paramStep == 6)
	{
		RRParam = param("RR-6-MM");
	}
	else if (paramStep != 1)
	{
		myThreadedLogger.Error("Unsupported step: " + to_string(paramStep));
		return;
	}

	info_t Z1000Info = Fetch(forecastTime, Z1000Level, ZParam, forecastType, false);
	info_t Z850Info = Fetch(forecastTime, Z850Level, ZParam, forecastType, false);
	info_t T850Info = Fetch(forecastTime, Z850Level, TParam, forecastType, false);
	info_t NInfo = Fetch(forecastTime, NLevel, NParams, forecastType, false);
	info_t TInfo = Fetch(forecastTime, T2Level, TParam, forecastType, false);
	info_t CloudInfo = Fetch(forecastTime, NLevel, CloudParam, forecastType, false);
	info_t KindexInfo = Fetch(forecastTime, NLevel, KindexParam, forecastType, false);
	info_t RRInfo = Fetch(forecastTime, forecastLevel, RRParam, forecastType, false);

	forecast_time nextTimeStep = forecastTime;
	nextTimeStep.ValidDateTime().Adjust(myTargetInfo->Time().StepResolution(), paramStep);

	info_t NextRRInfo = Fetch(nextTimeStep, forecastLevel, RRParam, forecastType, false);

	/*
	 * Sometimes we cannot find data for either time with the current forecast step.
	 *
	 * This happens for example with EC when the forecast step changes at forecast hour 90.
	 * At that hour the forecast step is 1, so current RR is fetched from hour 90 as parameter
	 * RR-1-MM. Hour 91 does not exist so that data is unavailable. In the database we have data
	 * for hour 93, but that is parameter RR-3-MM. As both precipitation parameters have to be
	 * of the same aggregation period, we have re-fetch both.
	 *
	 * This same thing happens at forecast hour 144 when step changes from 3h --> 6h.
	 */

	if (!RRInfo || !NextRRInfo)
	{
		if (paramStep == 1)
		{
			paramStep = 3;
			RRParam = param("RR-3-MM");
		}
		else if (paramStep == 3)
		{
			paramStep = 6;
			RRParam = param("RR-6-MM");
		}
		else
		{
			myThreadedLogger.Error("Precipitation data not found");
			return;
		}

		RRInfo = Fetch(forecastTime, forecastLevel, RRParam, forecastType, false);
		nextTimeStep = forecastTime;
		nextTimeStep.ValidDateTime().Adjust(myTargetInfo->Time().StepResolution(), paramStep);

		NextRRInfo = Fetch(nextTimeStep, forecastLevel, RRParam, forecastType, false);
	}

	if (!Z1000Info || !Z850Info || !T850Info || !NInfo || !TInfo || !CloudInfo || !KindexInfo || !RRInfo || !NextRRInfo)
	{
		myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " +
		                         static_cast<string>(forecastLevel));
		return;
	}

	string deviceType = "CPU";

	// Precipitation limits copied from TEE_Hsade.F

	double RRLimit1 = 0.01;
	double RRLimit2 = 0.1;
	double RRLimit3 = 1.;
	double RRLimit4 = 3.;

	if (paramStep == 3)
	{
		assert(myTargetInfo->Time().StepResolution() == kHourResolution);

		RRLimit1 = 0.1;
		RRLimit2 = 0.5;
		RRLimit3 = 2.;
		RRLimit4 = 4.;
	}
	else if (paramStep == 6)
	{
		assert(myTargetInfo->Time().StepResolution() == kHourResolution);

		RRLimit1 = 0.2;
		RRLimit2 = 1.;
		RRLimit3 = 4.;
		RRLimit4 = 8.;
	}

	LOCKSTEP(myTargetInfo, Z1000Info, Z850Info, T850Info, NInfo, TInfo, CloudInfo, KindexInfo, RRInfo, NextRRInfo)
	{
		double N = NInfo->Value();
		double T = TInfo->Value();
		double Z1000 = Z1000Info->Value();
		double T850 = T850Info->Value();
		double Z850 = Z850Info->Value();
		double cloud = CloudInfo->Value();
		double kindex = KindexInfo->Value();
		double RR = RRInfo->Value();
		double nextRR = NextRRInfo->Value();

		if (IsMissingValue({N, T, Z1000, Z850, T850, cloud, kindex, RR, nextRR}))
		{
			continue;
		}

		double reltopo = metutil::RelativeTopography_(1000, 850, Z1000, Z850);

		double rain = 0;       // default, no rain
		double cloudType = 1;  // default

		// from rain intensity determine WaWa-code

		if (nextRR > RRLimit4 && RR > RRLimit4)
		{
			rain = 65;
		}
		else if (nextRR > RRLimit3 && RR > RRLimit3)
		{
			rain = 63;
		}
		else if (nextRR > RRLimit2 && RR > RRLimit2)
		{
			rain = 61;
		}
		else if (nextRR > RRLimit1 && RR > RRLimit1)
		{
			rain = 60;
		}

		// cloud code determines cloud type

		N *= 100;

		if (cloud == 3307)
		{
			cloudType = 2;  // sade alapilvesta
		}
		else if (cloud == 2307 && N > 70)
		{
			cloudType = 2;
		}
		else if (cloud == 3604)
		{
			cloudType = 3;  // sade paksusta pilvesta
		}
		else if (cloud == 3309 || cloud == 2303 || cloud == 2302 || cloud == 1309 || cloud == 1303 || cloud == 1302)
		{
			cloudType = 4;  // kuuropilvi
		}

		// Ukkoset
		T850 = T850 - himan::constants::kKelvin;

		if (cloudType == 2 && T850 < -9) cloudType = 5;  // lumisade

		if (cloudType == 4)
		{
			if (kindex >= 37)
				cloudType = 45;  // ukkossade

			else if (kindex >= 27)
				cloudType = 35;  // ukkossade
		}

		// from here HSADE1-N

		if (rain >= 60 && rain <= 65)
		{
			if (cloudType == 3)  // Jatkuva sade
			{
				if (reltopo < 1288)
				{
					rain = rain + 10;  // Lumi
				}
				else if (reltopo > 1300)
				{
					// rain = rain;   // Vesi
				}
				else
				{
					rain = 68;  // Räntä
				}
			}
			else if (cloudType == 45)  // Kuuroja + voimakasta ukkosta
			{
				if (reltopo < 1285)
				{
					if (rain >= 63)  // Lumikuuroja
					{
						rain = 86;
					}
					else
					{
						rain = 85;
					}
				}
				else
				{
					rain = 97;  // Kesällä ukkosta
				}
			}
			else if (cloudType == 35)  // Kuuroja + ukkosta
			{
				if (reltopo < 1285)
				{
					if (rain >= 63)  // Lumikuuroja
					{
						rain = 86;
					}
					else
					{
						rain = 85;
					}
				}
				else
				{
					rain = 95;  // Kesällä ukkosta
				}
			}
			else if (cloudType == 4)  // Kuuroja - ukkosta
			{
				if (reltopo < 1285)
				{
					if (rain >= 63)
					{
						rain = 86;
					}
					else
					{
						rain = 85;
					}
				}
				else
				{
					if (rain >= 63)  // Vesikuuroja
					{
						rain = 82;
					}
					else
					{
						rain = 80;
					}
				}
			}
			else if (cloudType == 2)  // Tihkua
			{
				if (rain <= 61)  // Sademäärä ei saa olla suuri
				{
					if (reltopo < 1288)
					{
						rain = 78;  // Lumikiteitä
					}
					else
					{
						rain = rain - 10;  // Tihkua
					}
				}
			}
			else if (cloudType == 5)  // Lumisadetta alapilvistä
			{
				rain = rain + 10;
			}
			else  // Hetkellisen sateen virhe, siis poutaa
			{
				rain = 0;
			}

			if (reltopo >= 1289)  // Lopuksi jäätävä sade
			{
				if (rain >= 60 && rain <= 61 && T <= 270.15)
				{
					rain = 66;
				}
				else if (rain >= 62 && rain <= 65 && T <= 270.15)
				{
					rain = 67;
				}
				else if (rain >= 50 && rain <= 51 && T <= 273.15)
				{
					rain = 56;
				}
				else if (rain >= 52 && rain <= 55 && T <= 273.15)
				{
					rain = 57;
				}
			}
		}

		myTargetInfo->Value(rain);
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
