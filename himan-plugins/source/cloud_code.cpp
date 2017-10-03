/**
 * @file cloud_code.cpp
 *
 */

#include "cloud_code.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "metutil.h"

using namespace std;
using namespace himan::plugin;

const string itsName("cloud_code");

cloud_code::cloud_code()
{
	itsLogger = logger(itsName);
}

void cloud_code::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetParams({param("CLDSYM-N", 328, 0, 6, 8)});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void cloud_code::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	// Required source parameters

	const param TParam("T-K");
	const params RHParam = {param("RH-0TO1"), param("RH-PRCNT")};
	const params NParams = {param("N-0TO1"), param("N-PRCNT")};
	const param KParam("KINDEX-N");

	level T0mLevel(himan::kHeight, 0, "HEIGHT");
	level NKLevel(himan::kHeight, 0, "HEIGHT");
	level RH850Level(himan::kPressure, 850, "PRESSURE");
	level RH700Level(himan::kPressure, 700, "PRESSURE");
	level RH500Level(himan::kPressure, 500, "PRESSURE");

	auto myThreadedLogger = logger(itsName + "Thread #" + to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                      static_cast<string>(forecastLevel));

	info_t T0mInfo = Fetch(forecastTime, T0mLevel, TParam, forecastType, false);
	info_t NInfo = Fetch(forecastTime, NKLevel, NParams, forecastType, false);
	info_t KInfo = Fetch(forecastTime, NKLevel, KParam, forecastType, false);
	info_t T850Info = Fetch(forecastTime, RH850Level, TParam, forecastType, false);
	info_t RH850Info = Fetch(forecastTime, RH850Level, RHParam, forecastType, false);
	info_t RH700Info = Fetch(forecastTime, RH700Level, RHParam, forecastType, false);
	info_t RH500Info = Fetch(forecastTime, RH500Level, RHParam, forecastType, false);

	if (!T0mInfo || !NInfo || !KInfo || !T850Info || !RH850Info || !RH700Info || !RH500Info)
	{
		myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " +
		                         static_cast<string>(forecastLevel));
		return;
	}

	string deviceType = "CPU";

	double percentMultiplier = 1.0;

	if (RH500Info->Param().Unit() != kPrcnt)
	{
		itsLogger.Info("RH parameter unit not kPrcnt, assuming 0 .. 1");
		percentMultiplier = 100.0;
	}

	// Special case for hirlam
	if (myTargetInfo->Producer().Name() == "HL2MTA")
	{
		percentMultiplier = 100.0;
	}

	LOCKSTEP(myTargetInfo, T0mInfo, NInfo, KInfo, T850Info, RH850Info, RH700Info, RH500Info)
	{
		double T0m = T0mInfo->Value();
		double N = NInfo->Value();
		double kIndex = KInfo->Value();
		double T850 = T850Info->Value();
		double RH850 = RH850Info->Value();
		double RH700 = RH700Info->Value();
		double RH500 = RH500Info->Value();

		if (IsMissingValue({T0m, N, kIndex, T850, RH850, RH700, RH500}))
		{
			continue;
		}

		// error codes from fortran
		int cloudCode = 704;

		int lowConvection = metutil::LowConvection_(T0m, T850);

		// data comes as 0..1 instead of 0-100%
		N *= 100;

		RH500 *= percentMultiplier;
		RH700 *= percentMultiplier;
		RH850 *= percentMultiplier;

		if (N > 90)
		// Jos N = 90…100 % (pilvistä), niin
		{
			if (RH500 > 65)
			{
				cloudCode = 3502;
			}
			else
			{
				cloudCode = 3306;
			}

			if (RH700 > 80)
			{
				cloudCode = 3405;
			}

			if (RH850 > 60)
			{
				if (RH700 > 80)
				{
					cloudCode = 3604;
					myTargetInfo->Value(cloudCode);
					continue;
				}
				else
				{
					cloudCode = 3307;
				}
			}

			// jos RH500 > 65, tulos 3502 (yläpilvi)
			// ellei, niin tulos 3306 (yhtenäinen alapilvi)
			// Jos kuitenkin RH700 > 80, tulos 3405 (keskipilvi)
			// Jos kuitenkin RH850 > 60, niin
			// jos RH700 > 80, tulos 3604 (paksut kerrospilvet) tyyppi = 3 (sade) > ulos
			// ellei, niin tulos 3307 (alapilvi) tyyppi = 2

			if (kIndex > 25)
			{
				cloudCode = 3309;
				//	cloudType = 4;
			}

			else if (kIndex > 20)
			{
				cloudCode = 2303;
				//	cloudType = 4;
			}

			else if (kIndex > 15)
			{
				cloudCode = 2302;
				//	cloudType = 4;
			}
			else if (lowConvection == 1)
			{
				cloudCode = 2303;
				//	cloudType = 4;
			}
			/*
			Jos kIndex > 25, niin tulos 3309 (iso kuuropilvi), tyyppi 4
			Jos kIndex > 20, niin tulos 2303 (korkea konvektiopilvi), tyyppi 4
			Jos kIndex > 15, niin tulos 2302 (konvektiopilvi), tyyppi 4
			Jos lowConvection = 1, niin tulos 2303 (korkea konvektiopilvi), tyyppi 4
			*/
		}
		else if (N > 50)
		// Jos N = 50…90 % (puolipilvistä tai pilvistä), niin
		{
			if (RH500 > 65)
			{
				cloudCode = 2501;
			}

			else
			{
				cloudCode = 2305;
			}

			if (RH700 > 80)
			{
				cloudCode = 2403;
			}

			if (RH850 > 80)
			{
				cloudCode = 2307;

				//	if ( N > 70 )
				//		cloudType = 2;
			}
			/*	jos RH500 > 65, tulos 2501 (cirrus)
			    ellei, niin tulos 2305 (stratocumulus)
			    Jos kuitenkin RH700 > 80, tulos 2403 (keskipilvi)
			    Jos kuitenkin RH850 > 80, tulos 2307 (matala alapilvi)
			        ja jos N > 70 %, tyyppi 2
			*/

			if (kIndex > 25)
			{
				cloudCode = 3309;
				//	cloudType = 4;
			}

			else if (kIndex > 20)
			{
				cloudCode = 2303;
				//	cloudType = 4;
			}

			else if (kIndex > 15)
			{
				cloudCode = 2302;
				//	cloudType = 4;
			}
			else if (lowConvection == 1)
			{
				cloudCode = 2303;
				//	cloudType = 4;
			}
			/*
			Jos kIndex > 25, niin tulos 3309 (iso kuuropilvi), tyyppi 4
			Jos kIndex > 20, niin tulos 2303 (korkea konvektiopilvi), tyyppi 4
			Jos kIndex > 15, niin tulos 2302 (konvektiopilvi), tyyppi 4
			Jos lowConvection = 1, niin tulos 2303 (korkea konvektiopilvi), tyyppi 4
			*/
		}
		else if (N > 10)
		// Jos N = 10… 50 % (hajanaista pilvisyyttä)
		{
			if (RH500 > 60)
			{
				cloudCode = 1501;
			}

			else
			{
				cloudCode = 1305;
			}

			if (RH700 > 70)
			{
				cloudCode = 1403;
			}

			if (RH850 > 80)
			{
				cloudCode = 1305;
			}

			//	jos RH500 > 60, niin tulos 1501 (ohutta cirrusta), muuten tulos 1305 (alapilveä)
			//	Jos RH700 > 70, tulos 1403 (keskipilveä)
			//	Jos RH850 > 80, tulos 1305 (alapilveä)

			if (kIndex > 25)
			{
				cloudCode = 1309;
				//	cloudType = 4;
			}

			else if (kIndex > 20)
			{
				cloudCode = 1303;
				//	cloudType = 4;
			}

			else if (kIndex > 15)
			{
				cloudCode = 1302;
				//	cloudType = 4;
			}

			else if (lowConvection == 2)
			{
				cloudCode = 1301;
			}

			else if (lowConvection == 1)
			{
				cloudCode = 1303;
				//	cloudType = 4;
			}

			/*
			Jos kIndex > 25, niin tulos 1309 (korkea kuuropilvi), tyyppi 4
			Jos kIndex > 20, niin tulos 1303 (korkea konvektiopilvi), tyyppi 4
			Jos kIndex > 15, niin tulos 1302 (konvektiopilvi), tyyppi 4
			Jos lowConvection = 2, niin tulos 1301 (poutapilvi)
			Jos lowConvection = 1, niin tulos 1303 (korkea konvektiopilvi), tyyppi 4
			*/
		}
		else
		// Jos N 0…10 %
		{
			// tulos 0. Jos lowConvection = 1, tulos 1303, tyyppi 4
			cloudCode = 0;
			if (lowConvection == 1)
			{
				cloudCode = 1303;
				//	cloudType = 4;
			}
		}

		myTargetInfo->Value(cloudCode);
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
