/**
 * @file turbulence.cpp
 */
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "plugin_factory.h"
#include "turbulence.h"
#include "util.h"

using namespace std;
using namespace himan::plugin;

turbulence::turbulence()
{
	itsLogger = logger("turbulence");
}

void turbulence::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter properties
	 * - name PARM_NAME, this name is found from neons. For example: T-K
	 * - univ_id UNIV_ID, newbase-id, ie code table 204
	 * - grib1 id must be in database
	 * - grib2 descriptor X'Y'Z, http://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_table4-2.shtml
	 *
	 */

	// param theRequestedParam(PARM_NAME, UNIV_ID, GRIB2DISCIPLINE, GRIB2CATEGORY, GRIB2NUMBER);
	param TI("TI-S2", 1164, 0, 19, 22);
	param TI2("TI2-S2", 1209, 0, 19, 22);
	// If this param is also used as a source param for other calculations
	// (like for example dewpoint, relative humidity), unit should also be
	// specified

	TI.Unit(kS2);
	TI2.Unit(kS2);

	SetParams({TI, TI2});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void turbulence::Calculate(info_t myTargetInfo, unsigned short threadIndex)
{
	/*
	 * Required source parameters
	 */

	const param UParam("U-MS");
	const param VParam("V-MS");
	const param HParam("HL-M");
	// ----

	// Current time and level as given to this thread

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	level prevLevel, nextLevel;

	prevLevel = level(myTargetInfo->Level());
	prevLevel.Value(myTargetInfo->Level().Value() - 1);
	prevLevel.Index(prevLevel.Index() - 1);

	nextLevel = level(myTargetInfo->Level());
	nextLevel.Value(myTargetInfo->Level().Value() + 1);
	nextLevel.Index(nextLevel.Index() + 1);

	auto myThreadedLogger = logger("turbulence_pluginThread #" + to_string(threadIndex));

	myThreadedLogger.Debug("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                       static_cast<string>(forecastLevel));

	info_t UInfo, VInfo, HInfo, prevUInfo, prevVInfo, prevHInfo, nextUInfo, nextVInfo, nextHInfo;

	prevHInfo = Fetch(forecastTime, prevLevel, HParam, forecastType, false);
	prevUInfo = Fetch(forecastTime, prevLevel, UParam, forecastType, false);
	prevVInfo = Fetch(forecastTime, prevLevel, VParam, forecastType, false);

	nextHInfo = Fetch(forecastTime, nextLevel, HParam, forecastType, false);
	nextUInfo = Fetch(forecastTime, nextLevel, UParam, forecastType, false);
	nextVInfo = Fetch(forecastTime, nextLevel, VParam, forecastType, false);

	HInfo = Fetch(forecastTime, forecastLevel, HParam, forecastType, false);
	UInfo = Fetch(forecastTime, forecastLevel, UParam, forecastType, false);
	VInfo = Fetch(forecastTime, forecastLevel, VParam, forecastType, false);

	if (!(prevHInfo && prevUInfo && prevVInfo && nextHInfo && nextUInfo && nextVInfo && HInfo && UInfo && VInfo))
	{
		myThreadedLogger.Info("Skipping step " + to_string(forecastTime.Step()) + ", level " +
		                      static_cast<string>(forecastLevel));
		return;
	}

	// If calculating for hybrid levels, A/B vertical coordinates must be set
	// (copied from source)
	myTargetInfo->ParamIndex(0);
	SetAB(myTargetInfo, HInfo);

	myTargetInfo->ParamIndex(1);
	SetAB(myTargetInfo, HInfo);

	string deviceType = "CPU";

	double Di = myTargetInfo->Grid()->Di();
	double Dj = myTargetInfo->Grid()->Dj();
	point firstPoint = myTargetInfo->Grid()->FirstPoint();

	size_t Ni = myTargetInfo->Grid()->Ni();
	size_t Nj = myTargetInfo->Grid()->Nj();

	vector<double> dx(Nj, MissingDouble());
	vector<double> dy(Ni, MissingDouble());

	for (size_t i = 0; i < Ni; ++i)
	{
		dy[i] = util::LatitudeLength(0) * Dj / 360;
	}

	for (size_t j = 0; j < Nj; ++j)
	{
		dx[j] = util::LatitudeLength(firstPoint.Y() + double(j) * Dj) * Di / 360;
	}

	pair<matrix<double>, matrix<double>> gradU = util::CentralDifference(UInfo->Data(), dx, dy);
	pair<matrix<double>, matrix<double>> gradV = util::CentralDifference(VInfo->Data(), dx, dy);

	LOCKSTEP(myTargetInfo, UInfo, VInfo, HInfo, prevUInfo, prevVInfo, prevHInfo, nextUInfo, nextVInfo, nextHInfo)
	{
		size_t index = myTargetInfo->LocationIndex();
		double U = UInfo->Value();
		double V = VInfo->Value();
		double H = HInfo->Value();
		double prevU = prevUInfo->Value();
		double prevV = prevVInfo->Value();
		double prevH = prevHInfo->Value();
		double nextU = nextUInfo->Value();
		double nextV = nextVInfo->Value();
		double nextH = nextHInfo->Value();

		if (IsMissingValue({U, V, H, prevU, prevV, prevH, nextU, nextV, nextH}))
		{
			continue;
		}

		// Precalculation of wind shear, deformation and convergence
		double WS = sqrt(pow((prevU + U + nextU) / 3, 2) + pow((prevV + V + nextV) / 3, 2));
		double VWS = sqrt(pow((nextU - prevU) / (nextH - prevH), 2) + pow((nextV - prevV) / (nextH - prevH), 2));
		double DEF = sqrt(pow(get<0>(gradU).At(index) - get<1>(gradV).At(index), 2) +
		                  pow(get<0>(gradV).At(index) + get<1>(gradU).At(index), 2));
		double CVG = -get<0>(gradU).At(index) - get<1>(gradV).At(index);

		// Calculate scaling factor
		double S = MissingDouble();
		double ScaleMax = 40;
		double ScaleMin = 10;
		if (WS >= ScaleMax)
		{
			S = 1;
		}
		else if (WS >= ScaleMin && WS < ScaleMax)
		{
			S = WS / ScaleMax;
		}
		else
		{
			S = ScaleMin / ScaleMax;
		}

		// Calculation of turbulence indices
		double TI = S * VWS * DEF;
		double TI2 = S * VWS * (DEF + CVG);

		// return result
		myTargetInfo->ParamIndex(0);
		myTargetInfo->Value(TI);

		myTargetInfo->ParamIndex(1);
		myTargetInfo->Value(TI2);
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
