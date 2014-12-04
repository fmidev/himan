/**
 * @file fog.cpp
 *
 * @date Jul 3, 2013
 * @author peramaki
 */

#include "fog.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "level.h"
#include "forecast_time.h"

using namespace std;
using namespace himan::plugin;

const string itsName("fog");

fog::fog()
{
	itsClearTextFormula = "FOG = (DT2M-TGround> -0.3 && FF10M < 5) ? 607 : 0";
	itsLogger = logger_factory::Instance()->GetLog(itsName);

}

void fog::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetParams({param("FOGSYM-N", 334, 0, 6, 8)});

	Start();

}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void fog::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{

	// Required source parameters

	//2m dewpoint
	//10m wind speed
	//"platform" temperature

	const params groundParam = { param("T-K"), param("TG-K") };
	const param dewParam("TD-C");
	const param windParam("FF-MS");
	
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

	const level h2m(himan::kHeight, 2, "HEIGHT");
	const level h10m(himan::kHeight, 10, "HEIGHT");

	auto myThreadedLogger = logger_factory::Instance()->GetLog(itsName + "Thread #" + boost::lexical_cast<string> (threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	myThreadedLogger->Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));

	info_t groundInfo = Fetch(forecastTime, ground, groundParam, false);
	info_t dewInfo = Fetch(forecastTime, h2m, dewParam, false);
	info_t windInfo = Fetch(forecastTime, h10m, windParam, false);

	if (!groundInfo || !dewInfo || !windInfo)
	{
		myThreadedLogger->Warning("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", level " + static_cast<string> (forecastLevel));
		return;
	}

	string deviceType = "CPU";

	LOCKSTEP(myTargetInfo, groundInfo, dewInfo, windInfo)
	{

		double dt2m = dewInfo->Value();
		double wind10m = windInfo->Value();
		double tGround = groundInfo->Value();

		if (tGround == kFloatMissing || dt2m == kFloatMissing || wind10m == kFloatMissing)
		{
			continue;
		}

		double fog = 0;

		if (dt2m-tGround > -0.3 && wind10m < 5 )
		{
			fog = 607;
		}
		
		myTargetInfo->Value(fog);
	}
	
	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data().MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data().Size()));

}