/**
 * @file absolute_humidity.cpp
 *
 * Plug-in to calculate total humidity
 *
 */

#include "absolute_humidity.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"

using namespace std;
using namespace himan::plugin;

const string itsName("absolute_humidity");

absolute_humidity::absolute_humidity()
{
	itsLogger = logger(itsName);
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

	info_t RhoInfo = Fetch(forecastTime, forecastLevel, RhoParam, forecastType, false);
	info_t RainInfo = Fetch(forecastTime, forecastLevel, RainParam, forecastType, false);
	info_t SnowInfo = Fetch(forecastTime, forecastLevel, SnowParam, forecastType, false);
	info_t GraupelInfo = Fetch(forecastTime, forecastLevel, GraupelParam, forecastType, false);

	if (!RhoInfo || !RainInfo || !SnowInfo || !GraupelInfo)
	{
		myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " +
		                         static_cast<string>(forecastLevel));
		return;
	}

	SetAB(myTargetInfo, RhoInfo);

	// Calculate on CPU
	string deviceType = "CPU";

	auto& target = VEC(myTargetInfo);

	for (auto&& tup : zip_range(target, VEC(RhoInfo), VEC(RainInfo), VEC(SnowInfo), VEC(GraupelInfo)))
	{
		double& result = tup.get<0>();
		const double Rho = tup.get<1>();
		const double Rain = tup.get<2>();
		const double Snow = tup.get<3>();
		const double Graupel = tup.get<4>();

		// If mixing ratio is negative use 0.0 kg/kg
		// instead.
		result = Rho * fmax((Rain + Snow + Graupel), 0.0);
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
