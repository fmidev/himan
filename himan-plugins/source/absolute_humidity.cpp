/**
 * @file absolute_humidity.cpp
 *
 * Plug-in to calculate total humidity
 *
 * @date Mar 27, 2014
 * @author Tack
 */

#include "absolute_humidity.h"
#include "forecast_time.h"
#include "level.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace himan::plugin;

const string itsName("absolute_humidity");

absolute_humidity::absolute_humidity()
{
	itsClearTextFormula = "???";
	itsLogger = unique_ptr<logger>(logger_factory::Instance()->GetLog(itsName));
}

void absolute_humidity::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetParams({param("ABSH-KGM3", 1192, 0, 1, 18)});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void absolute_humidity::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	// Required source parameters (Density from plug-in density; rain, snow and graupel from Harmonie model output)

	const param RhoParam("RHO-KGM3");      // Density in kg/m3
	const param RainParam("RRI-KGM2");     // Large Scale precipitation in kg/m2
	const param SnowParam("SNRI-KGM2");    // Large scale snow accumulation in kg/m2
	const param GraupelParam("GRI-KGM2");  // Graupel precipitation in kg/m2

	auto myThreadedLogger =
	    logger_factory::Instance()->GetLog(itsName + "Thread #" + boost::lexical_cast<string>(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger->Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                       static_cast<string>(forecastLevel));

	info_t RhoInfo = Fetch(forecastTime, forecastLevel, RhoParam, forecastType, false);
	info_t RainInfo = Fetch(forecastTime, forecastLevel, RainParam, forecastType, false);
	info_t SnowInfo = Fetch(forecastTime, forecastLevel, SnowParam, forecastType, false);
	info_t GraupelInfo = Fetch(forecastTime, forecastLevel, GraupelParam, forecastType, false);

	if (!RhoInfo || !RainInfo || !SnowInfo || !GraupelInfo)
	{
		myThreadedLogger->Warning("Skipping step " + boost::lexical_cast<string>(forecastTime.Step()) + ", level " +
		                          static_cast<string>(forecastLevel));
		return;
	}

	SetAB(myTargetInfo, RhoInfo);

	// Calculate on CPU
	string deviceType = "CPU";

	LOCKSTEP(myTargetInfo, RhoInfo, RainInfo, SnowInfo, GraupelInfo)
	{
		double Rho = RhoInfo->Value();
		double Rain = RainInfo->Value();
		double Snow = SnowInfo->Value();
		double Graupel = GraupelInfo->Value();

		// Check if mixing ratio for rain is not missing
		if (Rho == kFloatMissing || Rain == kFloatMissing || Snow == kFloatMissing || Graupel == kFloatMissing)
		{
			continue;
		}

		// Calculate absolute humidity if mixing ratio is not missing. If mixing ratio is negative use 0.0 kg/kg
		// instead.
		double absolute_humidity = Rho * fmax((Rain + Snow + Graupel), 0.0);

		myTargetInfo->Value(absolute_humidity);
	}

	myThreadedLogger->Info("[" + deviceType + "] Missing values: " +
	                       boost::lexical_cast<string>(myTargetInfo->Data().MissingCount()) + "/" +
	                       boost::lexical_cast<string>(myTargetInfo->Data().Size()));
}
