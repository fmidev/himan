/**
 * @file precipitation_rate.cpp
 *
 * Template for future plugins.
 *
 */

#include "precipitation_rate.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"

using namespace std;
using namespace himan::plugin;

const string itsName("precipitation_rate");

precipitation_rate::precipitation_rate()
{
	itsLogger = logger(itsName);
}

void precipitation_rate::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	// First parameter - rain, second parameter - snow/solid precipitation

	SetParams({param("RRI-KGM2", 1171, 0, 1, 65), param("RSI-KGM2", 1193, 0, 1, 66)});

	Start<float>();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void precipitation_rate::Calculate(shared_ptr<info<float>> myTargetInfo, unsigned short threadIndex)
{
	// define quotients in formulas for rain rate and solid precipitation rate as constants
	const float rain_rate_factor = 1000.0f / 0.072f;
	const float rain_rate_exponent = 1.0f / 0.880f;
	const float snow_rate_factor = 1000.0f / 0.200f;
	const float snow_rate_exponent = 1.0f / 0.900f;

	// Required source parameters (Density from plug-in density; rain, snow and graupel from Harmonie model output)

	const param RhoParam("RHO-KGM3");          // Density in kg/m3
	const param RainParam("RAINMR-KGKG");      // Rain water mixing ratio in kg/kg
	const param SnowParam("SNOWMR-KGKG");      // Snow mixing ratio in kg/kg
	const param GraupelParam("GRAUPMR-KGKG");  // Graupel mixing ratio in kg/kg

	auto myThreadedLogger = logger(itsName + "Thread #" + to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                      static_cast<string>(forecastLevel));

	auto RhoInfo = Fetch<float>(forecastTime, forecastLevel, RhoParam, forecastType, false);
	auto RainInfo = Fetch<float>(forecastTime, forecastLevel, RainParam, forecastType, false);
	auto SnowInfo = Fetch<float>(forecastTime, forecastLevel, SnowParam, forecastType, false);
	auto GraupelInfo = Fetch<float>(forecastTime, forecastLevel, GraupelParam, forecastType, false);

	if (!RhoInfo || !RainInfo || !SnowInfo || !GraupelInfo)
	{
		itsLogger.Warning("Skipping step " + static_cast<string>(forecastTime.Step()) + ", level " +
		                  static_cast<string>(forecastLevel));
		return;
	}

	SetAB(myTargetInfo, RhoInfo);

	string deviceType = "CPU";

	myTargetInfo->Index<param>(0);
	auto& targetRain = VEC(myTargetInfo);

	myTargetInfo->Index<param>(1);
	auto& targetSolid = VEC(myTargetInfo);

	for (auto&& tup : zip_range(targetRain, targetSolid, VEC(RhoInfo), VEC(RainInfo), VEC(SnowInfo), VEC(GraupelInfo)))
	{
		float& rain = tup.get<0>();
		float& solid = tup.get<1>();
		float Rho = tup.get<2>();
		float Rain = tup.get<3>();
		float Snow = tup.get<4>();
		float Graupel = tup.get<5>();

		// Calculate rain rate if mixing ratio is not missing. If mixing ratio is negative use 0.0 kg/kg instead.

		if (!IsMissingValue({Rho, Rain}))
		{
			rain = powf(Rho * fmaxf(Rain, 0.0) * rain_rate_factor, rain_rate_exponent);
		}

		// Calculate solid precipitation rate if mixing ratios are not missing. If sum of mixing ratios is negative use
		// 0.0 kg/kg instead.

		if (!IsMissingValue({Rho, Snow, Graupel}))
		{
			solid = powf(Rho * fmaxf((Snow + Graupel), 0.0) * snow_rate_factor, snow_rate_exponent);
		}
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
