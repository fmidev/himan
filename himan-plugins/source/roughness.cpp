/**
 * @file roughness.cpp
 *
 * Template for calculation of surface roughness from HIRLAM data.
 *
 * @date Mar 27, 2014
 * @author Tack
 */

#include "roughness.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "level.h"
#include "forecast_time.h"

using namespace std;
using namespace himan::plugin;

const string itsName("roughness");

roughness::roughness()
{
	itsClearTextFormula = "roughness = terrain roughness + surface roughness";

	itsLogger = logger_factory::Instance()->GetLog(itsName);
}

void roughness::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);
	
	SetParams({param("SR-M", 283, 2, 0, 1)});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void roughness::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	const param RoughTParam("SR-M"); // Surface roughness terrain contribution
	const param RoughVParam("SRMOM-M"); // Surface roughness vegetation contribution

	auto myThreadedLogger = logger_factory::Instance()->GetLog(itsName + "Thread #" + boost::lexical_cast<string> (threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	myThreadedLogger->Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));

	info_t RoughTInfo = Fetch(forecastTime, forecastLevel, RoughTParam, false);
	info_t RoughVInfo = Fetch(forecastTime, forecastLevel, RoughVParam, false);

	if (!RoughTInfo || !RoughVInfo)
	{
		myThreadedLogger->Warning("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", level " + static_cast<string> (forecastLevel));
		return;
	}

	string deviceType = "CPU";

	LOCKSTEP(myTargetInfo, RoughTInfo, RoughVInfo)
	{

		double RoughT = RoughTInfo->Value();
		double RoughV = RoughVInfo->Value();

		if (RoughT == kFloatMissing || RoughV == kFloatMissing )
		{
			continue;
		}

		RoughT+=RoughV;

		myTargetInfo->Value(RoughT);

	}

	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data().MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data().Size()));
}
