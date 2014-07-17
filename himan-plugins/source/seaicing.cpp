/**
 * @file seaicing.cpp
 *
 *  Created on: Jan 03, 2013
 *  @author aaltom
 */

#include "seaicing.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "level.h"
#include "forecast_time.h"

using namespace std;
using namespace himan::plugin;

seaicing::seaicing()
{
	itsClearTextFormula = "SeaIcing = FF * ( -0.35 -T2m ) / ( 1 + 0.3 * ( T0 + 0.35 ))";

	itsLogger = logger_factory::Instance()->GetLog("seaicing");

}

void seaicing::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to seaicing
	 */

	SetParams({param("ICING-N", 480, 0, 0, 2)});

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
  const param FfParam("FF-MS"); // 10 meter wind
  const level FfLevel(himan::kHeight, 10, "HEIGHT");

  level ground;

  // this will come back to us
  if ( itsConfiguration->SourceProducer().Id() == 131)
  {
       ground = level(himan::kGndLayer, 0, "GNDLAYER");
  }
  else
  {
       ground = level(himan::kHeight, 0, "HEIGHT");
  }


	auto myThreadedLogger = logger_factory::Instance()->GetLog("seaicingThread #" + boost::lexical_cast<string> (theThreadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	myThreadedLogger->Info("Calculating time " + static_cast<string>(*forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));

	info_t TInfo = Fetch(forecastTime, TLevel, TParam, false);
	info_t TgInfo = Fetch(forecastTime, ground, TParam, false);
	info_t FfInfo = Fetch(forecastTime, FfLevel, FfParam, false);

	if (!TInfo || !TgInfo || !FfInfo)
	{
		myThreadedLogger->Warning("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", level " + static_cast<string> (forecastLevel));
		return;
	}

	string deviceType = "CPU";

	LOCKSTEP(myTargetInfo, TInfo, TgInfo, FfInfo)
	{
		double T = TInfo->Value();
		double Tg = TgInfo->Value();
		double Ff = FfInfo->Value();

		if (T == kFloatMissing || Tg == kFloatMissing || Ff == kFloatMissing)
		{
			myTargetInfo->Value(-10);
			continue;
		}

		double seaIcing;
		double TBase = 273.15;

		T = T - TBase;
		Tg = Tg - TBase;

		if (Tg < -2 )
		{
			seaIcing = -10;
		}
		else
		{
			seaIcing = Ff * ( -0.35 -T ) / ( 1 + 0.3 * ( Tg + 0.35 ));

			if (seaIcing > 100)
			{
				seaIcing = 100;
			}
		}

		myTargetInfo->Value(seaIcing);
	}

	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data()->MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data()->Size()));

}
