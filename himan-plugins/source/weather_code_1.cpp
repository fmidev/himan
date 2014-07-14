/**
 * @file weather_code_1
 *
 *
 * @date Apr 10, 2013
 * @author partio, peramaki, aalto
 */

#include "weather_code_1.h"
#include "logger_factory.h"
#include "metutil.h"
#include <boost/lexical_cast.hpp>
#include "level.h"
#include "forecast_time.h"

using namespace std;
using namespace himan::plugin;

const string itsName("weather_code_1");

// Required source parameters

const himan::param ZParam("Z-M2S2");
const himan::params NParams({himan::param("N-0TO1"), himan::param("N-PRCNT")});
const himan::param TParam("T-K");
const himan::param CloudParam("CLDSYM-N");
const himan::params PrecParams({himan::param("RR-1-MM"), himan::param("RRR-KGM2")});
const himan::param KindexParam("KINDEX-N");
const himan::param RRParam("RR-1-MM");

// ..and their levels
himan::level Z1000Level(himan::kPressure, 1000, "PRESSURE");
himan::level Z850Level(himan::kPressure, 850, "PRESSURE");
himan::level T2Level(himan::kHeight, 2, "HEIGHT");
himan::level NLevel(himan::kHeight, 0, "HEIGHT");

weather_code_1::weather_code_1()
{
	itsClearTextFormula = "<algorithm>";
	itsLogger = logger_factory::Instance()->GetLog(itsName);

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

	auto myThreadedLogger = logger_factory::Instance()->GetLog(itsName + "Thread #" + boost::lexical_cast<string> (threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	myThreadedLogger->Info("Calculating time " + static_cast<string>(*forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));

	int paramStep = 1;

	info_t Z1000Info = Fetch(forecastTime, Z1000Level, ZParam, false);
	info_t Z850Info = Fetch(forecastTime, Z850Level, ZParam, false);
	info_t T850Info = Fetch(forecastTime, Z850Level, TParam, false);
	info_t NInfo = Fetch(forecastTime, NLevel, NParams, false);
	info_t TInfo = Fetch(forecastTime, T2Level, TParam, false);
	info_t CloudInfo = Fetch(forecastTime, NLevel, CloudParam, false);
	info_t KindexInfo = Fetch(forecastTime, NLevel, KindexParam, false);
	info_t RRInfo = Fetch(forecastTime, forecastLevel, RRParam, false);

	forecast_time nextTimeStep = forecastTime;
	nextTimeStep.ValidDateTime()->Adjust(myTargetInfo->Time().StepResolution(), paramStep);

	info_t NextRRInfo = Fetch(nextTimeStep, forecastLevel, RRParam, false);

	if (!Z1000Info || !Z850Info || !T850Info || !NInfo || !TInfo || !CloudInfo || !KindexInfo || !RRInfo || !NextRRInfo)
	{
		myThreadedLogger->Warning("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", level " + static_cast<string> (forecastLevel));
		return;
	}

	string deviceType = "CPU";

	LOCKSTEP(myTargetInfo,Z1000Info,Z850Info,T850Info,NInfo,TInfo,CloudInfo,KindexInfo,RRInfo,NextRRInfo)
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

		double rain = 0; // default, no rain
		double cloudType = 1; // default

		// from rain intensity determine WaWa-code

		if (nextRR > 0.01 && RR > 0.01 )
		{
			rain = 60;
		}
		if (nextRR > 0.1 && RR > 0.1 )
		{
			rain = 61;
		}
		if (nextRR > 1 && RR > 1 )
		{
			rain = 63;
		}
		if (nextRR > 3 && RR > 3)
		{
			rain = 65;
		}

		// cloud code determines cloud type

		N *= 100;

		if (cloud == 3307 )
		{
			cloudType = 2;  // sade alapilvesta
		}
		else if (cloud == 2307 && N > 70 )
		{
			cloudType = 2;
		}
		else if (cloud == 3604)
		{
			cloudType = 3;	// sade paksusta pilvesta
		}
		else if (cloud == 3309 || cloud == 2303 || cloud == 2302
			 || cloud == 1309 || cloud == 1303 || cloud == 1302)
		{
		   cloudType = 4; 	// kuuropilvi
		}

		// Ukkoset
		T850 = T850 - himan::constants::kKelvin;

		if ( cloudType == 2 && T850 < -9 )
			cloudType = 5;  // lumisade

		if ( cloudType == 4 )
		{
			if (kindex >= 37)
				cloudType = 45;  // ukkossade

			else if (kindex >= 27)
				cloudType = 35; // ukkossade
		}


		// from here HSADE1-N

		if (rain >= 60 && rain <= 65)
		{
			if (cloudType == 3) // Jatkuva sade
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
			else if (cloudType == 45) // Kuuroja + voimakasta ukkosta
			{
				if (reltopo < 1285)
				{

					if (rain >= 63) //Lumikuuroja
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
			else if (cloudType == 35)   // Kuuroja + ukkosta
			{
				if (reltopo < 1285)
				{

					if (rain >= 63)   // Lumikuuroja
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
			else if (cloudType == 4) // Kuuroja - ukkosta
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
					if (rain >= 63) // Vesikuuroja
					{
						rain = 82;
					}
					else
					{
						rain = 80;
					}
				}
			}
			else if (cloudType == 2)   // Tihkua
			{
				if (rain <= 61) // Sademäärä ei saa olla suuri
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
			else if (cloudType == 5) //Lumisadetta alapilvistä
			{
				rain = rain + 10;
			}
			else // Hetkellisen sateen virhe, siis poutaa
			{
				rain = 0;
			}


			if (reltopo >= 1289) // Lopuksi jäätävä sade
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

	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data()->MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data()->Size()));

}