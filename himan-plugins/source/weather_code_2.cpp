/**
 * @file weather_code_2.cpp
 *
 */

#include "weather_code_2.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "metutil.h"

using namespace std;
using namespace himan::plugin;

weather_code_2::weather_code_2()
{
	itsLogger = logger("weather_code_2");
}

void weather_code_2::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetParams({param("ILSAA1-N", 350)});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void weather_code_2::Calculate(shared_ptr<info> myTargetInfo, unsigned short theThreadIndex)
{
	// Required source parameters
	// new parameters used...
	param PrecformParam("PRECFORM-N");
	param TotalPrecParam("RRR-KGM2");
	params TotalCloudCoverParam = {param("N-0TO1"), param("N-PRCNT")};
	param LowCloudCoverParam("NL-PRCNT");
	param MedCloudCoverParam("NM-PRCNT");
	param HighCloudCoverParam("NH-PRCNT");
	param FogParam("FOGSYM-N");
	param CloudParam("CLDSYM-N");

	// parameters to check convection
	const param TParam("T-K");
	const param KParam("KINDEX-N");

	level HLevel(himan::kHeight, 0, "HEIGHT");

	// levels to check convection
	level T0mLevel(himan::kHeight, 0, "HEIGHT");
	level RH850Level(himan::kPressure, 850, "PRESSURE");

	auto myThreadedLogger = logger("weather_code_2Thread #" + to_string(theThreadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                      static_cast<string>(forecastLevel));

	info_t CloudInfo = Fetch(forecastTime, HLevel, CloudParam, forecastType, false);
	info_t PrecformInfo = Fetch(forecastTime, HLevel, PrecformParam, forecastType, false);
	info_t TotalPrecInfo = Fetch(forecastTime, HLevel, TotalPrecParam, forecastType, false);
	info_t TotalCloudCoverInfo = Fetch(forecastTime, HLevel, TotalCloudCoverParam, forecastType, false);
	info_t LowCloudCoverInfo = Fetch(forecastTime, HLevel, LowCloudCoverParam, forecastType, false);
	info_t MedCloudCoverInfo = Fetch(forecastTime, HLevel, MedCloudCoverParam, forecastType, false);
	info_t HighCloudCoverInfo = Fetch(forecastTime, HLevel, HighCloudCoverParam, forecastType, false);
	info_t FogInfo = Fetch(forecastTime, HLevel, FogParam, forecastType, false);
	info_t T0mInfo = Fetch(forecastTime, T0mLevel, TParam, forecastType, false);
	info_t T850Info = Fetch(forecastTime, RH850Level, TParam, forecastType, false);
	info_t KInfo = Fetch(forecastTime, HLevel, KParam, forecastType, false);

	if (!CloudInfo || !PrecformInfo || !TotalPrecInfo || !TotalCloudCoverInfo || !LowCloudCoverInfo ||
	    !MedCloudCoverInfo || !HighCloudCoverInfo || !FogInfo || !T0mInfo || !T850Info || !KInfo)
	{
		myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " +
		                         static_cast<string>(forecastLevel));
		return;
	}

	string deviceType = "CPU";

	LOCKSTEP(myTargetInfo, CloudInfo, PrecformInfo, TotalPrecInfo, TotalCloudCoverInfo, LowCloudCoverInfo,
	         MedCloudCoverInfo, HighCloudCoverInfo, FogInfo, T0mInfo, T850Info, KInfo)
	{
		double precForm = PrecformInfo->Value();
		double totalPrec = TotalPrecInfo->Value();
		double totalCC = TotalCloudCoverInfo->Value();
		double lowCC = LowCloudCoverInfo->Value();
		double medCC = MedCloudCoverInfo->Value();
		double highCC = HighCloudCoverInfo->Value();
		double fog = FogInfo->Value();

		// convection input parameters
		double T0m = T0mInfo->Value();
		double kIndex = KInfo->Value();
		double T850 = T850Info->Value();

		// for thunder
		double cloud = CloudInfo->Value();

		// derived parameters
		double precType, fogIntensity, thunderProb;

		// output parameter
		double weather_symbol;

		if (IsMissingValue({cloud, totalPrec, totalCC, lowCC, medCC, highCC, fog, T0m, kIndex, T850, cloud}))
		{
			continue;
		}

		weather_symbol = 0;
		precType = rain_type(kIndex, T0m, T850);    // calculate rain type index
		fogIntensity = (fog == 607 ? 1 : 0);        // calculate fog intesity index
		thunderProb = thunder_prob(kIndex, cloud);  // calculate thunder probability

		// weather algorithm goes here

		// precipitation below 0.02 mm/h is assumed dry weather
		if (totalPrec <= 0.02)
		{
			// !!!Logical error, weather_symbol = 10 will always be overwritten in the else clause (in smartmet script,
			// fixed here by else if clause)!!!
			if (fogIntensity == 1)
				weather_symbol = 10;  // mist: two horizontal lines
			else if (fogIntensity == 2)
				weather_symbol = 40;  // fog: three horizontal lines
			else
			{
				// Total cloud cover less than 11%
				if (totalCC <= 0.1) weather_symbol = 0;  // clear: sun (moon and/or stars)
				// !!!Attention, N-0TO1 and NX-PRCNT are not percent. They're between 0 and 1!!!
				else if ((lowCC + medCC) <= 0.10 && totalCC > 0.10)
					weather_symbol = 1;  // thin high clouds: the sun and transparent cloud
				else if ((lowCC + medCC) > 0.10 && totalCC > 0.10 && totalCC <= 0.30)
					weather_symbol = 2;  // almost clear: the sun and the (small) light cloud
				else if ((lowCC + medCC) > 0.10 && totalCC > 0.30 && totalCC <= 0.60)
					weather_symbol = 3;  // Partly cloudy: the sun and the light cloud
				else if ((lowCC + medCC) > 0.10 && totalCC > 0.60 && totalCC <= 0.80)
					weather_symbol = 4;  // almost overcast: the sun and the (large) light cloud
				else if ((lowCC + medCC) > 0.10 && totalCC > 0.80)
					weather_symbol = 5;  // Cloudy: light cloud
			}
		}
		// rainfall of > 0.02 mm/h
		else if (precForm == 0)  // drizzling rain
		{
			if (totalPrec > 0.02 && totalPrec <= 0.03)
				weather_symbol = 51;  // weak drizzle: for example a dark cloud, and one rain drop
			else if (totalPrec > 0.03)
				weather_symbol = 53;  // drizzle: for example a dark cloud, and two rain drops
		}
		else if (precForm == 1)  // rainfall
		{
			// logical error for prectype == 2 will always give thunderstorm
			if (totalPrec <= 0.4 && precType == 1)
				weather_symbol = 61;  // slight rain: a dark cloud and one rain drop
			else if (totalPrec > 0.4 && totalPrec <= 2.0 && precType == 1)
				weather_symbol = 63;  // moderate rain: a dark cloud, and two rain drops
			else if (totalPrec > 2.0 && precType == 1)
				weather_symbol = 65;  // heavy rain: a dark cloud, and three rain drops
			else if (totalPrec <= 0.4 && precType == 2)
				weather_symbol = 80;  // Poor water: deaf Sun and half dark cloud droplets and lines
			else if (totalPrec > 0.4 && totalPrec <= 7.0 && precType == 2)
				weather_symbol = 81;  // moderate to hard water deaf: a dark cloud droplets and lines +
			else if (totalPrec > 7.0 && precType == 2)
				weather_symbol = 82;  // very heavy rain showers: CB cloud droplets and lines +
			/*
			 * Thunder so far determined by using the probability of thunder
			 * weak thunder when tn < 20%
			 * moderate thunder when the probability is 20-50%
			 * loud thunder when t > 50%
			 */
			if (precType == 2 && thunderProb > 20 && thunderProb <= 50)
				weather_symbol = 95;  // slight to moderate thunderstorms: the sun and the cb-cloud + lightning
			else if (precType == 2 && thunderProb > 50)
				weather_symbol = 97;  // heavy thunderstorms: cb-cloud + two lightning
		}
		else if (precForm == 2)  // sleet
		{
			if (totalPrec <= 0.4 && precType == 1)
				weather_symbol = 68;  // weak rain and snow: a dark cloud and flake + rain drop
			else if (totalPrec > 0.4 && precType == 1)
				weather_symbol = 69;  // moderate or heavy rain and snow: a dark cloud and two flakes + drops
			else if (totalPrec <= 0.4 && precType == 2)
				weather_symbol = 83;  // weak sleet showers: the sun and the dark side of the cloud and flake + drop
			else if (totalPrec > 0.4 && precType == 2)
				weather_symbol = 84;  // moderate or severe sleet showers: a dark cloud and 2x flake + drop
			/*
			 * Thunder so far determined by using the probability of thunder
			 * weak thunder when tn < 20%
			 * moderate thunder when the probability is 20-50%
			 * loud thunder when t > 50%
			 */
			if (precType == 2 && thunderProb > 20 && thunderProb <= 50)
				weather_symbol = 96;  // slight to moderate thunderstorms: the sun and the cb-cloud + lightning
			else if (precType == 2 && thunderProb > 50)
				weather_symbol = 99;  // heavy thunderstorms: cb-cloud + two lightning
		}
		else if (precForm == 3)  // snowfall
		{
			if (totalPrec <= 0.4 && precType == 1)
				weather_symbol = 71;  // light snow: a dark cloud and snowflakes
			else if (totalPrec > 0.4 && totalPrec <= 1.5 && precType == 1)
				weather_symbol = 73;  // moderate snow showers: a dark cloud, and 2 flakes
			else if (totalPrec > 1.5 && precType == 1)
				weather_symbol = 75;  // dense Snow: a dark cloud, and 3 flakes
			else if (totalPrec <= 0.4 && precType == 2)
				weather_symbol = 85;  // weak snow showers: the sun and the dark side of the cloud and lines + flakes
			else if (totalPrec > 0.4 && precType == 2)
				weather_symbol = 73;  // moderate or severe snow showers: a dark cloud lines and flakes
			/*
			 * Thunder so far determined by using the probability of thunder
			 * weak thunder when tn < 20%
			 * moderate thunder when the probability is 20-50%
			 * loud thunder when t > 50%
			 */
			if (precType == 2 && thunderProb > 20 && thunderProb <= 50)
				weather_symbol = 96;  // slight to moderate thunderstorms: the sun and the cb-cloud + lightning
			else if (precType == 2 && thunderProb > 50)
				weather_symbol = 99;  // heavy thunderstorms: cb-cloud + two lightning
		}
		else if (precForm == 4)  // freezing drizzle rain
		{
			if (totalPrec > 0.02 && totalPrec <= 0.03)
				weather_symbol = 56;  // Light Freezing Drizzle: for example, a dark cloud, and the dotted curved line
			else if (totalPrec > 0.03)
				weather_symbol = 57;  // freezing drizzle: for example, a dark cloud, and two curved line in a comma
		}
		else if (precForm == 5)  // sleet
		{
			if (totalPrec <= 0.4)
				weather_symbol = 66;  // Light Freezing Rain: a dark cloud and a drop of the arc line
			else
				weather_symbol = 67;  // moderate or heavy freezing rain: a dark cloud, and two drops of the arc line
			/*
			 * Thunder so far determined by using the probability of thunder
			 * weak thunder when tn < 20%
			 * moderate thunder when the probability is 20-50%
			 * loud thunder when t > 50%
			 */
			if (precType == 2 && thunderProb > 20 && thunderProb <= 50)
				weather_symbol = 96;  // slight to moderate thunderstorms: the sun and the cb-cloud + lightning
			else if (precType == 2 && thunderProb > 50)
				weather_symbol = 99;  // heavy thunderstorms: cb-cloud + two lightning
		}
		else if (precForm == 6 || precForm == 7)  // hail
		{
			/*
			 * Thunder so far determined by using the probability of thunder
			 * weak thunder when tn < 20%
			 * moderate thunder when the probability is 20-50%
			 * loud thunder when t > 50%
			 */
			if (precType == 2 && thunderProb > 20 && thunderProb <= 50)
				weather_symbol = 96;  // slight to moderate thunderstorms: the sun and the cb-cloud + lightning
			else if (precType == 2 && thunderProb > 50)
				weather_symbol = 95;  // heavy thunderstorms: cb-cloud + two lightning
		}

		myTargetInfo->Value(weather_symbol);
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}

double weather_code_2::rain_type(double kIndex, double T0m, double T850)
{
	double rain_type;
	if (kIndex > 15)
		rain_type = 2;  // check for convection
	else if (metutil::LowConvection_(T0m, T850) == 0)
		rain_type = 1;  // check for shallow convection
	else
		rain_type = 2;

	return rain_type;
}

double weather_code_2::thunder_prob(double kIndex, double cloud)
{
	double thunder_prob = 0;  // initialize thunder probability to 0%
	if (cloud == 3309 || cloud == 2303 || cloud == 2302 || cloud == 1309 || cloud == 1303 || cloud == 1302)
	{
		if (kIndex >= 37)
			thunder_prob = 60;  // heavy thunder, set thunder probability to a value over 50% (to be replaced by a more
		                        // scientific way to determine thunder probability in the future)
		else if (kIndex >= 27)
			thunder_prob = 40;  // thunder, set thunder probability to a value between 30% and 50% (to be replaced by a
		                        // more scientific way to determine thunder probability in the future)
	}

	return thunder_prob;
}
