/**
 * @file seaicing.cpp
 *
 */

#include "seaicing.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"

using namespace std;
using namespace himan::plugin;

seaicing::seaicing() : global(false)
{
	itsLogger = logger("seaicing");
}

void seaicing::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	if (itsConfiguration->Exists("global") && itsConfiguration->GetValue("global") == "true")
	{
		SetParams({param("SSICING-N", 10059, 0, 0, 2)});
		global = true;
	}
	else
	{
		// By default baltic sea
		SetParams({param("ICING-N", 480, 0, 0, 2)});
		global = false;
	}

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void seaicing::Calculate(shared_ptr<info> myTargetInfo, unsigned short theThreadIndex)
{
	const params TParam = {param("T-K"), param("TG-K")};
	const level TLevel(himan::kHeight, 2, "HEIGHT");
	const param FfParam("FF-MS");  // 10 meter wind
	const level FfLevel(himan::kHeight, 10, "HEIGHT");

	level ground;
	double saltinessIndex = 0.35;

	if (global)
	{
		saltinessIndex = 1.5;
	}

	if (itsConfiguration->SourceProducer().Id() == 131 || itsConfiguration->SourceProducer().Id() == 134)
	{
		ground = level(himan::kGroundDepth, 0, 7);
	}
	else
	{
		ground = level(himan::kHeight, 0, "HEIGHT");
	}

	auto myThreadedLogger = logger("seaicingThread #" + to_string(theThreadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
						  static_cast<string>(forecastLevel));

	info_t TInfo = Fetch(forecastTime, TLevel, TParam, forecastType, false);
	info_t TgInfo = Fetch(forecastTime, ground, TParam, forecastType, false);
	info_t FfInfo = Fetch(forecastTime, FfLevel, FfParam, forecastType, false);

	if (!TInfo || !TgInfo || !FfInfo)
	{
		myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " +
		                         static_cast<string>(forecastLevel));
		return;
	}

	string deviceType = "CPU";

	LOCKSTEP(myTargetInfo, TInfo, TgInfo, FfInfo)
	{
		double T = TInfo->Value();
		double Tg = TgInfo->Value();
		double Ff = FfInfo->Value();

		double seaIcing = MissingDouble();
		double TBase = 273.15;

		T = T - TBase;
		Tg = Tg - TBase;

		if (Tg < -2)
		{
			seaIcing = 0;
		}
		else
		{
			seaIcing = Ff * (-saltinessIndex - T) / (1 + 0.3 * (Tg + saltinessIndex));

			// Change values to index
			// Index by Antonios Niros: Vessel icing forecast and services: further development and perspectives.

			if (seaIcing <= 0)
			{  // No icing
				seaIcing = 0;
			}
			else if (seaIcing > 0 && seaIcing < 22.4)
			{  // Light icing ja icing rate <0.7cm/h
				seaIcing = 1;
			}
			else if (seaIcing >= 22.4 && seaIcing < 53.3)
			{  // Moderate icing ja icing rate between 0.7cm/h-2cm/h
				seaIcing = 2;
			}
			else if (seaIcing >= 53.3 && seaIcing < 83)
			{  //  Heavy icing ja icing rate between 2.0cm/h-4.0cm/h
				seaIcing = 3;
			}
			else if (seaIcing >= 83)
			{  // Extreme icing ja icing rate >4cm/h
				seaIcing = 4;
			}
		}

		myTargetInfo->Value(seaIcing);
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
