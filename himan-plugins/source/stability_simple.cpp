#include "stability_simple.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "metutil.h"
#include "util.h"

#include "fetcher.h"
#include "hitool.h"
#include "radon.h"
#include "writer.h"

using namespace std;
using namespace himan;
using namespace himan::plugin;

const himan::param KIParam("KINDEX-N");
const himan::param VTIParam("VTI-N");
const himan::param CTIParam("CTI-N");
const himan::param TTIParam("TTI-N");
const himan::param TParam("T-K");
const himan::param TDParam("TD-K");
const himan::level P850Level(himan::kPressure, 850);
const himan::level P700Level(himan::kPressure, 700);
const himan::level P500Level(himan::kPressure, 500);

/**
 * @brief Cross Totals Index
 *
 * http://glossary.ametsoc.org/wiki/Stability_index
 *
 * @param T500 Temperature of 500 hPa isobar in Kelvins
 * @param TD850 Dewpoint temperature of 850 hPa isobar in Kelvins
 * @return Index value (TD850 - T500)
 */

inline double CTI(double T500, double TD850)
{
	return TD850 - T500;
}

/**
 * @brief Vertical Totals Index
 *
 * http://glossary.ametsoc.org/wiki/Stability_index
 *
 * @param T850 Temperature of 850 hPa isobar in Kelvins
 * @param T500 Temperature of 500 hPa isobar in Kelvins
 * @return Index value (T850 - T500)
 */

inline double VTI(double T850, double T500)
{
	return T850 - T500;
}
/**
 * @brief Total Totals Index
 *
 * http://glossary.ametsoc.org/wiki/Stability_index
 *
 * @param T850 Temperature of 850 hPa isobar in Kelvins
 * @param T500 Temperature of 500 hPa isobar in Kelvins
 * @param TD850 Dewpoint temperature of 850 hPa isobar in Kelvins
 * @return Index value ( T850 - T500 ) + ( TD850 - T500 )
 */

inline double TTI(double T850, double T500, double TD850)
{
	return CTI(T500, TD850) + VTI(T850, T500);
}

/**
 * @brief K-Index
 *
 * @param T500 Temperature of 500 hPa isobar in Kelvins
 * @param T700 Temperature of 700 hPa isobar in Kelvins
 * @param T850 Temperature of 850 hPa isobar in Kelvins
 * @param TD700 Dewpoint temperature of 700 hPa isobar in Kelvins
 * @param TD850 Dewpoint temperature of 850 hPa isobar in Kelvins
 * @return Index value
 */

inline double KI(double T850, double T700, double T500, double TD850, double TD700)
{
	return (T850 - T500 + TD850 - (T700 - TD700)) - himan::constants::kKelvin;
}

stability_simple::stability_simple()
{
	itsLogger = logger("stability_simple");
}
void stability_simple::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetParams({KIParam, CTIParam, VTIParam, TTIParam});

	Start();
}

void stability_simple::Calculate(shared_ptr<info<double>> myTargetInfo, unsigned short theThreadIndex)
{
	auto myThreadedLogger = logger("stability_simpleThread #" + to_string(theThreadIndex));

	const forecast_time forecastTime = myTargetInfo->Time();
	const level forecastLevel = myTargetInfo->Level();
	const forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                      static_cast<string>(forecastLevel));

	string deviceType = "CPU";

	auto T850Info = Fetch(forecastTime, P850Level, TParam, forecastType, false);
	auto T700Info = Fetch(forecastTime, P700Level, TParam, forecastType, false);
	auto T500Info = Fetch(forecastTime, P500Level, TParam, forecastType, false);
	auto TD850Info = Fetch(forecastTime, P850Level, TDParam, forecastType, false);
	auto TD700Info = Fetch(forecastTime, P700Level, TDParam, forecastType, false);

	if (!T850Info || !T700Info || !T500Info || !TD850Info || !TD700Info)
	{
		return;
	}

	myTargetInfo->Find<param>(KIParam);
	auto& ki = VEC(myTargetInfo);

	myTargetInfo->Find<param>(CTIParam);
	auto& cti = VEC(myTargetInfo);

	myTargetInfo->Find<param>(VTIParam);
	auto& vti = VEC(myTargetInfo);

	myTargetInfo->Find<param>(TTIParam);
	auto& tti = VEC(myTargetInfo);

	const auto& t850 = VEC(T850Info);
	const auto& td850 = VEC(TD850Info);
	const auto& t700 = VEC(T700Info);
	const auto& td700 = VEC(TD700Info);
	const auto& t500 = VEC(T500Info);

	for (size_t i = 0; i < tti.size(); i++)
	{
		const double T850 = t850[i];
		const double T700 = t700[i];
		const double T500 = t500[i];
		const double TD850 = td850[i];
		const double TD700 = td700[i];

		ki[i] = ::KI(T850, T700, T500, TD850, TD700);
		cti[i] = ::CTI(T500, TD850);
		vti[i] = ::VTI(T850, T500);
		tti[i] = ::TTI(T850, T500, TD850);
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing: " + to_string(util::MissingPercent(*myTargetInfo)) + "%");
}
