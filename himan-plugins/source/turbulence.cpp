/**
 * @file turbulence.cpp
 */
#include "turbulence.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "plugin_factory.h"
#include "util.h"

using namespace std;
using namespace himan::plugin;

#ifdef HAVE_CUDA
namespace turbulence_cuda
{
extern void Process(std::shared_ptr<const himan::plugin_configuration> conf,
                    std::shared_ptr<himan::info<float>> myTargetInfo);
}
#endif

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

	Start<float>();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void turbulence::Calculate(shared_ptr<info<float>> myTargetInfo, unsigned short threadIndex)
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

	string deviceType;

#ifdef HAVE_CUDA

	if (itsConfiguration->UseCuda())
	{
		deviceType = "GPU";

		turbulence_cuda::Process(itsConfiguration, myTargetInfo);
	}
	else
#endif
	{
		shared_ptr<info<float>> UInfo, VInfo, HInfo, prevUInfo, prevVInfo, prevHInfo, nextUInfo, nextVInfo, nextHInfo;

		prevHInfo = Fetch<float>(forecastTime, prevLevel, HParam, forecastType, false);
		prevUInfo = Fetch<float>(forecastTime, prevLevel, UParam, forecastType, false);
		prevVInfo = Fetch<float>(forecastTime, prevLevel, VParam, forecastType, false);

		nextHInfo = Fetch<float>(forecastTime, nextLevel, HParam, forecastType, false);
		nextUInfo = Fetch<float>(forecastTime, nextLevel, UParam, forecastType, false);
		nextVInfo = Fetch<float>(forecastTime, nextLevel, VParam, forecastType, false);

		HInfo = Fetch<float>(forecastTime, forecastLevel, HParam, forecastType, false);
		UInfo = Fetch<float>(forecastTime, forecastLevel, UParam, forecastType, false);
		VInfo = Fetch<float>(forecastTime, forecastLevel, VParam, forecastType, false);

		if (!(prevHInfo && prevUInfo && prevVInfo && nextHInfo && nextUInfo && nextVInfo && HInfo && UInfo && VInfo))
		{
			myThreadedLogger.Info("Skipping step " + static_cast<string>(forecastTime.Step()) + ", level " +
			                      static_cast<string>(forecastLevel));
			return;
		}

		// If calculating for hybrid levels, A/B vertical coordinates must be set
		// (copied from source)
		myTargetInfo->Index<param>(0);
		SetAB(myTargetInfo, HInfo);

		myTargetInfo->Index<param>(1);
		SetAB(myTargetInfo, HInfo);

		deviceType = "CPU";

		ASSERT(myTargetInfo->Grid()->Class() == kRegularGrid);

		auto gr = dynamic_pointer_cast<regular_grid>(myTargetInfo->Grid());

		const float Di = static_cast<float>(gr->Di());
		const float Dj = static_cast<float>(gr->Dj());
		point firstPoint = myTargetInfo->Grid()->FirstPoint();

		const size_t Ni = gr->Ni();
		const size_t Nj = gr->Nj();

		vector<float> dx, dy;

		bool jPositive;
		if (gr->ScanningMode() == kTopLeft)
		{
			jPositive = false;
		}
		else if (gr->ScanningMode() == kBottomLeft)
		{
			jPositive = true;
		}
		else
		{
			myThreadedLogger.Error("Grid not supported for CAT calculation.");
			himan::Abort();
		}

		switch (UInfo->Grid()->Type())
		{
			case kLambertConformalConic:
			{
				dx = vector<float>(Nj, Di);
				dy = vector<float>(Ni, Dj);
				break;
			};
			case kRotatedLatitudeLongitude:
				// When working in rotated space, first point must also be rotated
				firstPoint =
				    dynamic_pointer_cast<rotated_latitude_longitude_grid>(myTargetInfo->Grid())->Rotate(firstPoint);
				// fall through
			case kLatitudeLongitude:
			{
				dx = vector<float>(Nj, MissingFloat());
				dy = vector<float>(Ni, MissingFloat());
				float jPositiveFloat = jPositive ? 1.0f : -1.0f;

				for (size_t i = 0; i < Ni; ++i)
				{
					dy[i] = util::LatitudeLength(0.0f) * Dj / 360.0f;
				}

				for (size_t j = 0; j < Nj; ++j)
				{
					dx[j] = util::LatitudeLength(static_cast<float>(firstPoint.Y()) + float(j) * Dj * jPositiveFloat) *
					        Di / 360.0f;
				}
				break;
			}
			default:
			{
				myThreadedLogger.Error("Grid not supported for CAT calculation.");
				himan::Abort();
			}
		}

		pair<matrix<float>, matrix<float>> gradU = util::CentralDifference(UInfo->Data(), dx, dy, jPositive);
		pair<matrix<float>, matrix<float>> gradV = util::CentralDifference(VInfo->Data(), dx, dy, jPositive);

		LOCKSTEP(myTargetInfo, UInfo, VInfo, HInfo, prevUInfo, prevVInfo, prevHInfo, nextUInfo, nextVInfo, nextHInfo)
		{
			size_t index = myTargetInfo->LocationIndex();
			float U = UInfo->Value();
			float V = VInfo->Value();
			float prevU = prevUInfo->Value();
			float prevV = prevVInfo->Value();
			float prevH = prevHInfo->Value();
			float nextU = nextUInfo->Value();
			float nextV = nextVInfo->Value();
			float nextH = nextHInfo->Value();

			if (IsMissingValue({U, V, prevU, prevV, prevH, nextU, nextV, nextH}))
			{
				continue;
			}

			// Precalculation of wind shear, deformation and convergence
			float WS = sqrt(pow((prevU + U + nextU) / 3.0f, 2.0f) + pow((prevV + V + nextV) / 3.0f, 2.0f));
			float VWS =
			    sqrt(pow((nextU - prevU) / (nextH - prevH), 2.0f) + pow((nextV - prevV) / (nextH - prevH), 2.0f));
			float DEF = sqrt(pow(get<0>(gradU).At(index) - get<1>(gradV).At(index), 2.0f) +
			                 pow(get<0>(gradV).At(index) + get<1>(gradU).At(index), 2.0f));
			float CVG = -get<0>(gradU).At(index) - get<1>(gradV).At(index);

			// Calculate scaling factor
			float S;
			float ScaleMax = 40;
			float ScaleMin = 10;
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
			float TI = S * VWS * DEF;
			float TI2 = S * VWS * (DEF + CVG);

			// return result
			myTargetInfo->Index<param>(0);
			myTargetInfo->Value(TI);

			myTargetInfo->Index<param>(1);
			myTargetInfo->Value(TI2);
		}
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
