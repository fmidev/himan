/**
 * @file precipitation_rate.cpp
 *
 * Template for future plugins.
 *
 * @date Mar 14, 2014
 * @author Tack
 */

#include "precipitation_rate.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "level.h"
#include "forecast_time.h"

using namespace std;
using namespace himan::plugin;

const string itsName("precipitation_rate");

precipitation_rate::precipitation_rate()
{
	itsClearTextFormula = "???";

	itsLogger = logger_factory::Instance()->GetLog(itsName);

}

void precipitation_rate::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	// First parameter - rain, second parameter - snow/solid precipitation

	SetParams({param("RRI-KGM2", 1171, 0, 1, 65), param("RSI-KGM2", 1193, 0, 1, 66)});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void precipitation_rate::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{

	// define quotients in formulas for rain rate and solid precipitation rate as constants
	const double rain_rate_factor = 1000.0/0.072;
	const double rain_rate_exponent = 1.0/0.880;
	const double snow_rate_factor = 1000.0/0.200;
	const double snow_rate_exponent = 1.0/0.900;

	// Required source parameters (Density from plug-in density; rain, snow and graupel from Harmonie model output)

	const param RhoParam("RHO-KGM3");	// Density in kg/m3
	const param RainParam("RRI-KGM2");	// Large Scale precipitation in kg/m2
	const param SnowParam("SNRI-KGM2");	// Large scale snow accumulation in kg/m2
	const param GraupelParam("GRI-KGM2");	// Graupel precipitation in kg/m2

	auto myThreadedLogger = logger_factory::Instance()->GetLog(itsName + "Thread #" + boost::lexical_cast<string> (threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	myThreadedLogger->Info("Calculating time " + static_cast<string>(*forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));

	info_t RhoInfo = Fetch(forecastTime, forecastLevel, RhoParam, false);
	info_t RainInfo = Fetch(forecastTime, forecastLevel, RainParam, false);
	info_t SnowInfo = Fetch(forecastTime, forecastLevel, SnowParam, false);
	info_t GraupelInfo = Fetch(forecastTime, forecastLevel, GraupelParam, false);
	
	if (!RhoInfo || !RainInfo || !SnowInfo || !GraupelInfo)
	{
		itsLogger->Warning("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", level " + static_cast<string> (forecastLevel));
		return;
	}

	SetAB(myTargetInfo, RhoInfo);
		
	string deviceType = "CPU";

	LOCKSTEP(myTargetInfo, RhoInfo, RainInfo, SnowInfo, GraupelInfo)
	{

		double Rho = RhoInfo->Value();
		double Rain = RainInfo->Value();
		double Snow = SnowInfo->Value();
		double Graupel = GraupelInfo->Value();

		// Calculate rain rate if mixing ratio is not missing. If mixing ratio is negative use 0.0 kg/kg instead.

		if (Rho != kFloatMissing && Rain != kFloatMissing)
		{

			double rain_rate = pow(Rho * fmax(Rain, 0.0) * rain_rate_factor, rain_rate_exponent);

			assert(rain_rate == rain_rate);  // Checking NaN (note: assert() is defined only in debug builds)

			myTargetInfo->ParamIndex(0);

			myTargetInfo->Value(rain_rate);
		}

		// Calculate solid precipitation rate if mixing ratios are not missing. If sum of mixing ratios is negative use 0.0 kg/kg instead.
		if (Rho != kFloatMissing && Snow != kFloatMissing && Graupel != kFloatMissing)
		{
			double sprec_rate = pow(Rho * fmax((Snow + Graupel), 0.0) * snow_rate_factor, snow_rate_exponent);

			assert(sprec_rate == sprec_rate); // Checking NaN (note: assert() is defined only in debug builds)

			myTargetInfo->ParamIndex(1);

			myTargetInfo->Value(sprec_rate);

		}
	}

	myThreadedLogger->Info("Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data().MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data().Size()));
}
