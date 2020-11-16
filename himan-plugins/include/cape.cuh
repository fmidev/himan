#pragma once

#include "himan_common.h"
#include "info.h"
#include "metutil.h"
#include "moisture.h"
#include "numerical_functions.h"

/*
 * Namespace CAPE holds common CAPE integration functions
 * that are called from CPU and GPU code.
 */

#define LINEAR himan::numerical_functions::interpolation::Linear<float>

const himan::param LCLTParam("LCL-K");
const himan::param LCLPParam("LCL-HPA");
const himan::param LCLZParam("LCL-M");
const himan::param LFCTParam("LFC-K");
const himan::param LFCPParam("LFC-HPA");
const himan::param LFCZParam("LFC-M");
const himan::param ELTParam("EL-K");
const himan::param ELPParam("EL-HPA");
const himan::param ELZParam("EL-M");
const himan::param LPLTParam("LPL-K");
const himan::param LPLPParam("LPL-HPA");
const himan::param LPLZParam("LPL-M");
const himan::param LastELTParam("EL-LAST-K");
const himan::param LastELPParam("EL-LAST-HPA");
const himan::param LastELZParam("EL-LAST-M");
const himan::param CAPEParam("CAPE-JKG");
const himan::param CAPE1040Param("CAPE1040-JKG");
const himan::param CAPE3kmParam("CAPE3KM-JKG");
const himan::param CINParam("CIN-JKG");
const himan::param PParam("P-HPA");
const himan::param TParam("T-K");
const himan::param ZParam("HL-M");
const himan::param RHParam("RH-PRCNT");

const double mucape_search_limit = 550.;         // hPa
const double mucape_maxima_search_limit = 650.;  // hPa

namespace CAPE
{
CUDA_DEVICE
inline himan::point GetPointOfIntersection(const himan::point& a1, const himan::point& a2, const himan::point& b1,
                                           const himan::point& b2)
{
	double x1 = a1.X(), x2 = a2.X(), x3 = b1.X(), x4 = b2.X();
	double y1 = a1.Y(), y2 = a2.Y(), y3 = b1.Y(), y4 = b2.Y();

	double d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

	himan::point null(himan::MissingDouble(), himan::MissingDouble());

	if (d == 0)
	{
		// parallel lines
		return null;
	}

	double pre = (x1 * y2 - y1 * x2);
	double post = (x3 * y4 - y3 * x4);

	// Intersection x & y
	double x = (pre * (x3 - x4) - (x1 - x2) * post) / d;
	double y = (pre * (y3 - y4) - (y1 - y2) * post) / d;

	if (x < fmin(x1, x2) - 1e5 || x > fmax(x1, x2) + 1e5 || x < fmin(x3, x4) - 1e5 || x > fmax(x3, x4) + 1e5)
	{
		return null;
	}

	if (y < fmin(y1, y2) - 1e5 || y > fmax(y1, y2) + 1e5 || y < fmin(y3, y4) - 1e5 || y > fmax(y3, y4) + 1e5)
	{
		return null;
	}

	return himan::point(x, y);
}

CUDA_DEVICE
inline float IntegrateEnteringParcel(float Tenv, float prevTenv, float Tparcel, float prevTparcel, float Zenv,
                                     float prevZenv)
{
	/*
	 *  We just entered CAPE or CIN zone.
	 *
	 *                                             Hybrid level n == Zenv
	 *                        This point is Tenv --> ======== <-- this point is Tparcel
	 *                                                \####/
	 *                                                 \##/
	 *                                                  \/  <-- This point is going to be new prevZenv that we get from
	 * intersectionWithZ.
	 *                                                  /\      At this point obviously Tenv = Tparcel.
	 *                                                 /  \
	 *          This line is the raising particle --> /    \ <-- This line is the environment temperature
	 *                                               ========
	 *                                             Hybrid level n+1 == prevZenv
	 *
	 *  We want to calculate only the upper triangle!
	 *
	 *  Summary:
	 *  1. Calculate intersection of lines in order to get the height of the point where Tparcel == Tenv. This point is
	 * going
	 *     to be the new prevZenv.
	 *  2. Calculate integral using dz = Zenv - prevZenv, for temperatures use the values from Hybrid level n.
	 */

	using himan::point;

	auto intersection = CAPE::GetPointOfIntersection(point(Tenv, Zenv), point(prevTenv, prevZenv), point(Tparcel, Zenv),
	                                                 point(prevTparcel, prevZenv));

	if (!(intersection.Y() == intersection.Y()))
	{
		return 0;
	}

	prevZenv = static_cast<float>(intersection.Y());

	float value = static_cast<float>(himan::constants::kG) * (Zenv - prevZenv) * ((Tparcel - Tenv) / Tenv);
	value = fminf(150.f, fmaxf(-150.f, value));

	ASSERT(!isnan(value) && !isinf(value));

	return value;
}

CUDA_DEVICE
inline void IntegrateLeavingParcel(float Tenv, float prevTenv, float Tparcel, float prevTparcel, float Penv,
                                   float prevPenv, float Zenv, float prevZenv, float& out_value, float& out_ELT,
                                   float& out_ELP, float& out_ELZ)
{
	/*
	 *  We just left CAPE or CIN zone.
	 *
	 *                                             Hybrid level n == Zenv
	 *                                               ========
	 *                                                 \  /
	 *                                                  \/  <-- This point is going to be new Zenv that we get from
	 * intersectionWithZ.
	 *                                                  /\      At this point obviously Tenv = Tparcel.
	 *                                                 /##\
	 *   This line is the environment temperature --> /####\ <-- this line is the raising particle
	 *                   This point is prevTenv -->  ========  <-- This point is prevTparcel
	 *                                             Hybrid level n+1 == prevZenv
	 *
	 *  We want to calculate only the lower triangle!
	 *
	 *  Summary:
	 *  1. Calculate intersection of lines in order to get the height of the point where Tparcel == Tenv. This point is
	 * going
	 *     to be the new Zenv.
	 *  2. Calculate integral using dz = ZenvNew - prevZenv, for temperatures use the values from Hybrid level n+1.
	 */

	out_value = 0;

	using himan::point;

	auto intersectionZ = CAPE::GetPointOfIntersection(point(Tenv, Zenv), point(prevTenv, prevZenv),
	                                                  point(Tparcel, Zenv), point(prevTparcel, prevZenv));

	if (!(intersectionZ.Y() == intersectionZ.Y()))
	{
		return;
	}

	auto intersectionP = CAPE::GetPointOfIntersection(point(Tenv, Penv), point(prevTenv, prevPenv),
	                                                  point(Tparcel, Penv), point(prevTparcel, prevPenv));

	if (!(intersectionP.X() == intersectionP.X()))
	{
		return;
	}

	Zenv = static_cast<float>(intersectionZ.Y());
	ASSERT(fabs(intersectionZ.X() - intersectionP.X()) < 1.);
	float value = static_cast<float>(himan::constants::kG) * (Zenv - prevZenv) * ((prevTparcel - prevTenv) / prevTenv);
	value = fminf(150.f, fmaxf(-150.f, value));

	ASSERT(!isnan(value) && !isinf(value));

	out_value = value;
	out_ELT = static_cast<float>(intersectionP.X());
	out_ELP = static_cast<float>(intersectionP.Y());
	out_ELZ = static_cast<float>(intersectionZ.Y());
}

CUDA_DEVICE
inline float IntegrateHeightAreaLeavingParcel(float Tenv, float prevTenv, float Tparcel, float prevTparcel, float Zenv,
                                              float prevZenv, float areaUpperLimit)
{
	/*
	 * Just left valid CAPE zone to a non-valid area.
	 *
	 * Note! Parcel is buoyant at both areas!
	 *
	 * In this example parcel is lifted to over 3km.
	 *
	 *       =========
	 *         \  |
	 *          \  \
	 *           \##\  <-- 3km height
	 *            \#|
	 *            |#|
	 *       =========
	 *
	 *
	 *  We want to calculate only the '#' area!
	 */

	float newTenv = LINEAR(areaUpperLimit, Zenv, prevZenv, Tenv, prevTenv);
	float newTparcel = LINEAR(areaUpperLimit, Zenv, prevZenv, Tparcel, prevTparcel);

	if (newTparcel <= newTenv)
	{
		// Tparcel has to be warmer than environment, otherwise no CAPE

		for (int i = 0; i < 20; i++)
		{
			areaUpperLimit -= 10;

			newTenv = LINEAR(areaUpperLimit, Zenv, prevZenv, Tenv, prevTenv);
			newTparcel = LINEAR(areaUpperLimit, Zenv, prevZenv, Tparcel, prevTparcel);

			if (areaUpperLimit <= prevZenv)
			{
				// Lower height reached upper height
				return 0;
			}
			else if (newTparcel > newTenv)
			{
				// Found correct height
				break;
			}
		}

		if (newTparcel <= newTenv)
		{
			// Unable to find the height where env temp is cold enough AND Tparcel is warmer than Tenv
			return 0;
		}
	}

	ASSERT(newTparcel >= newTenv);
	ASSERT(areaUpperLimit > prevZenv);

	float CAPE =
	    static_cast<float>(himan::constants::kG) * (areaUpperLimit - prevZenv) * ((prevTparcel - prevTenv) / prevTenv);

	CAPE = fminf(CAPE, 150.);

	ASSERT(CAPE >= 0.);
	ASSERT(CAPE <= 150);

	return CAPE;
}

CUDA_DEVICE
inline float IntegrateTemperatureAreaEnteringParcel(float Tenv, float prevTenv, float Tparcel, float prevTparcel,
                                                    float Zenv, float prevZenv, float areaColderLimit,
                                                    float areaWarmerLimit)
{
	/*
	 * Just entered valid CAPE zone from a non-valid area.
	 *
	 * Note! Parcel is buoyant at both areas!
	 *
	 * In this example we entered a cold cape area (-) from a warmer area (+).
	 *
	 *          ##########
	 *           \-----|
	 *            \----|
	 *             \---|
	 *              \++|
	 *               \+|
	 *          ##########
	 *
	 *
	 *  We want to calculate only the '-' area!
	 *
	 *  Summary:
	 *  1. Calculate the point where the env temperature crosses to cold area (ie. 263.15K). This point lies somewhere
	 *     between the two levels, and it's found with linear interpolation. The result should be new value for prevZ.
	 *     Note that the interpolation is done to the virtual temperatures, so that we don't have to interpolate
	 * pressure again!
	 *  2. Sometimes Tparcel is colder than Tenv at that height where Tenv crosser to colder area --> not in CAPE zone.
	 *     In that case we must find the first height where Tenv >= 263.15 and Tparcel >= Tenv.
	 *  3. Calculate integral using dz = Zenv - prevZenv, for temperatures use the values from Hybrid level n.
	 */

	float areaLimit;
	bool fromWarmerToCold = true;

	if (prevTenv > Tenv)
	{
		// Entering area from a warmer zone
		areaLimit = areaWarmerLimit;
	}
	else
	{
		// Entering area from a colder zone
		areaLimit = areaColderLimit;
		fromWarmerToCold = false;
	}

	float newPrevZenv = LINEAR(areaLimit, Tenv, prevTenv, Zenv, prevZenv);
	float newTparcel = LINEAR(newPrevZenv, Zenv, prevZenv, Tparcel, prevTparcel);

	if (newTparcel < areaLimit)
	{
		// Tparcel has to be warmer than environment, otherwise no CAPE

		for (int i = 0; i < 20; i++)
		{
			areaLimit += (fromWarmerToCold) ? -0.1f : 0.1f;

			newPrevZenv = LINEAR(areaLimit, Tenv, prevTenv, Zenv, prevZenv);
			newTparcel = LINEAR(newPrevZenv, Zenv, prevZenv, Tparcel, prevTparcel);

			if (newPrevZenv >= Zenv)
			{
				// Lower height reached upper height
				return 0;
			}
			else if (newTparcel >= areaLimit)
			{
				// Found correct height
				break;
			}
		}

		if (newTparcel <= areaLimit)
		{
			// Unable to find the height where env temp is cold enough AND Tparcel is warmer than Tenv
			return 0;
		}
	}

	ASSERT(Tparcel >= Tenv);
	ASSERT(Zenv >= newPrevZenv);

	float CAPE = static_cast<float>(himan::constants::kG) * (Zenv - newPrevZenv) * ((Tparcel - Tenv) / Tenv);
	CAPE = fminf(CAPE, 150.);

	ASSERT(Zenv >= prevZenv);
	ASSERT(CAPE >= 0.);
	ASSERT(CAPE <= 150.);

	return CAPE;
}

CUDA_DEVICE
inline float IntegrateTemperatureAreaLeavingParcel(float Tenv, float prevTenv, float Tparcel, float prevTparcel,
                                                   float Zenv, float prevZenv, float areaColderLimit,
                                                   float areaWarmerLimit)
{
	/*
	 * Just left valid CAPE zone to a non-valid area.
	 *
	 * Note! Parcel is buoyant at both areas!
	 *
	 *      ##########      ##########
	 *         \==|           \++|
	 *          \==\           \++\
	 *           \--\           \--\
	 *            \-|            \-|
	 *            |-|            |-|
	 *       ##########     ##########
	 *
	 *
	 *  We want to calculate only the '-' area!
	 */

	float areaLimit;
	bool fromColdToWarmer = true;

	if (prevTenv < Tenv)
	{
		// Entering to a warmer area
		areaLimit = areaWarmerLimit;
	}
	else
	{
		// Entering to a colder area
		areaLimit = areaColderLimit;
		fromColdToWarmer = false;
	}

	float newZenv = LINEAR(areaLimit, Tenv, prevTenv, Zenv, prevZenv);
	float newTparcel = LINEAR(newZenv, Zenv, prevZenv, Tparcel, prevTparcel);

	if (newTparcel <= areaLimit)
	{
		// Tparcel has to be warmer than environment, otherwise no CAPE

		for (int i = 0; i < 20; i++)
		{
			areaLimit += (fromColdToWarmer) ? -0.1f : 0.1f;

			newZenv = LINEAR(areaLimit, Tenv, prevTenv, Zenv, prevZenv);
			newTparcel = LINEAR(newZenv, Zenv, prevZenv, Tparcel, prevTparcel);

			if (newZenv <= prevZenv)
			{
				// Lower height reached upper height
				return 0;
			}
			else if (newTparcel >= areaLimit)
			{
				// Found correct height
				break;
			}
		}

		if (newTparcel <= areaLimit)
		{
			// Unable to find the height where env temp is cold enough AND Tparcel is warmer than Tenv
			return 0;
		}
	}

	ASSERT(Tparcel >= Tenv);
	ASSERT(newZenv <= Zenv);
	ASSERT(newZenv >= prevZenv);

	float CAPE = static_cast<float>(himan::constants::kG) * (Zenv - prevZenv) * ((newTparcel - areaLimit) / areaLimit);
	ASSERT(CAPE >= 0.);

	CAPE = fminf(CAPE, 150.);

	ASSERT(CAPE <= 150.);

	return CAPE;
}

CUDA_DEVICE
inline float CalcCAPE1040(float Tenv, float prevTenv, float Tparcel, float prevTparcel, float Penv, float prevPenv,
                          float Zenv, float prevZenv)
{
	float C = 0;

	ASSERT((Tenv == Tenv) && (Penv == Penv) && (Tparcel == Tparcel));

	if (Tparcel < Tenv && prevTparcel < prevTenv)
	{
		// No CAPE
		return C;
	}

	float coldColderLimit = 233.15f;
	float coldWarmerLimit = 263.15f;

	if (Tparcel > Tenv)
	{
		// Parcel is buoyant at current height

		if (Tenv >= coldColderLimit && Tenv <= coldWarmerLimit)
		{
			// Parcel is inside cold area at current height

			if (prevTenv > coldWarmerLimit || prevTenv < coldColderLimit)
			{
				// Entering cold cape area from either warmer or colder area
				C = CAPE::IntegrateTemperatureAreaEnteringParcel(Tenv, prevTenv, Tparcel, prevTparcel, Zenv, prevZenv,
				                                                 coldColderLimit, coldWarmerLimit);
			}
			else
			{
				// Firmly in the cold zone
				C = static_cast<float>(himan::constants::kG) * (Zenv - prevZenv) * ((Tparcel - Tenv) / Tenv);
				ASSERT(C >= 0.);
			}
		}
		else if ((prevTenv > coldColderLimit &&
		          prevTenv <
		              coldWarmerLimit)  // At previous height conditions were suitable (TODO: buoyancy is not checked!)
		         && (Tenv < coldColderLimit || Tenv > coldWarmerLimit))
		{
			// Current env temperature is too cold or too warm
			C = CAPE::IntegrateTemperatureAreaLeavingParcel(Tenv, prevTenv, Tparcel, prevTparcel, Zenv, prevZenv,
			                                                coldColderLimit, coldWarmerLimit);
		}
	}
	else if (prevTparcel >= prevTenv)
	{
		// No buoyancy anymore at current height, but
		// we HAD buoyancy: we just exited from a CAPE zone

		if (prevTenv >= coldColderLimit && prevTenv <= coldWarmerLimit)
		{
			/* Just left cold CAPE zone for an warmer or colder area */
			float CAPE, x1, x2, x3;
			CAPE::IntegrateLeavingParcel(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv, CAPE, x1,
			                             x2, x3);
			C = CAPE;
		}
	}

	return C;
}

CUDA_DEVICE
inline float CalcCAPE3km(float Tenv, float prevTenv, float Tparcel, float prevTparcel, float Penv, float prevPenv,
                         float Zenv, float prevZenv)
{
	float C = 0.;

	if (Tparcel > Tenv)
	{
		// Have buoyancy at current height

		if (Zenv <= 3000.)
		{
			if (prevTparcel >= prevTenv)
			{
				// Firmly in the zone
				C = static_cast<float>(himan::constants::kG) * (Zenv - prevZenv) * ((Tparcel - Tenv) / Tenv);
			}
			else
			{
				// Just entered CAPE zone
				C = CAPE::IntegrateEnteringParcel(Tenv, prevTenv, Tparcel, prevTparcel, Zenv, prevZenv);
			}
		}
		else if (prevZenv <= 3000.)
		{
			// Parcel has risen over 3km
			// Integrate from previous level to 3km (if parcel is buoyant the whole height)

			if (prevTparcel >= prevTenv)
			{
				C = CAPE::IntegrateHeightAreaLeavingParcel(Tenv, prevTenv, Tparcel, prevTparcel, Zenv, prevZenv, 3000);
			}
		}
	}
	else
	{
		// Exited CAPE zone, no buoyancy at this height

		if (prevTparcel >= prevTenv)
		{
			if (Zenv <= 3000.)
			{
				// Integrate from previous height to intersection
				float CAPE, x1, x2, x3;
				CAPE::IntegrateLeavingParcel(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv, CAPE,
				                             x1, x2, x3);
				C = CAPE;
			}

			else
			{
				// Integrate from previous height to 3km
				C = CAPE::IntegrateHeightAreaLeavingParcel(Tenv, prevTenv, Tparcel, prevTparcel, Zenv, prevZenv, 3000);
			}
		}
	}

	return C;
}

CUDA_DEVICE
inline void CalcCAPE(float Tenv, float prevTenv, float Tparcel, float prevTparcel, float Penv, float prevPenv,
                     float Zenv, float prevZenv, float& out_CAPE, float& out_ELT, float& out_ELP, float& out_ELZ)
{
	out_CAPE = 0.;
	out_ELP = himan::MissingFloat();
	out_ELT = himan::MissingFloat();
	out_ELZ = himan::MissingFloat();

	ASSERT((Tenv == Tenv) && (Penv == Penv) && (Tparcel == Tparcel));

	if (Tparcel < Tenv && prevTparcel < prevTenv)
	{
		// No CAPE
		return;
	}

	if (Tparcel >= Tenv && prevTparcel >= Tenv)
	{
		// We are fully in a CAPE zone
		out_CAPE = static_cast<float>(himan::constants::kG) * (Zenv - prevZenv) * ((Tparcel - Tenv) / Tenv);
	}
	else if (Tparcel >= Tenv && prevTparcel < prevTenv)
	{
		out_CAPE = CAPE::IntegrateEnteringParcel(Tenv, prevTenv, Tparcel, prevTparcel, Zenv, prevZenv);
	}
	else if (Tparcel < Tenv && prevTparcel >= prevTenv)
	{
		CAPE::IntegrateLeavingParcel(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv, out_CAPE,
		                             out_ELT, out_ELP, out_ELZ);
	}

	ASSERT(out_CAPE >= 0);
}

CUDA_DEVICE
inline float CalcCIN(float Tenv, float prevTenv, float Tparcel, float prevTparcel, float Penv, float prevPenv,
                     float Zenv, float prevZenv)
{
	if (Tparcel >= Tenv && prevTparcel >= prevTenv)
	{
		// No CIN
		return 0;
	}

	float cin = 0;

	if (Tparcel < Tenv && prevTparcel < prevTenv)
	{
		// We are fully in a CIN zone
		cin = static_cast<float>(himan::constants::kG) * (Zenv - prevZenv) * ((Tparcel - Tenv) / Tenv);
	}
	else if (Tparcel < Tenv && prevTparcel >= prevTenv)
	{
		cin = CAPE::IntegrateEnteringParcel(Tenv, prevTenv, Tparcel, prevTparcel, Zenv, prevZenv);
	}
	else if (Tparcel >= Tenv && prevTparcel < prevTenv)
	{
		float x1, x2, x3;
		CAPE::IntegrateLeavingParcel(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv, cin, x1, x2,
		                             x3);
	}

	return cin;
}

}  // namespace CAPE

#ifdef HAVE_CUDA
#include "cuda_helper.h"
#include "plugin_configuration.h"

typedef std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> cape_source;
typedef std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>, std::vector<std::vector<float>>>
    cape_multi_source;
typedef std::tuple<std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>,
                   std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>>
    CAPEdata;

namespace himan
{
namespace plugin
{
namespace cape_cuda
{
cape_multi_source GetNHighestThetaEValuesGPU(const std::shared_ptr<const plugin_configuration>& conf,
                                             std::shared_ptr<info<float>> myTargetInfo, int N);
std::vector<std::pair<std::vector<float>, std::vector<float>>> GetLFCGPU(
    const std::shared_ptr<const plugin_configuration>& conf, std::shared_ptr<info<float>> myTargetInfo,
    std::vector<float>& T, std::vector<float>& P, std::vector<float>& TenvLCL);
cape_source Get500mMixingRatioValuesGPU(std::shared_ptr<const plugin_configuration>& conf,
                                        std::shared_ptr<info<float>> myTargetInfo);
std::vector<float> GetCINGPU(const std::shared_ptr<const plugin_configuration>& conf,
                             std::shared_ptr<info<float>> myTargetInfo, const std::vector<float>& Tsource,
                             const std::vector<float>& Psource, const std::vector<float>& PLCL,
                             const std::vector<float>& PLFC, const std::vector<float>& ZLFC);
CAPEdata GetCAPEGPU(const std::shared_ptr<const plugin_configuration>& conf, std::shared_ptr<info<float>> myTargetInfo,
                    const std::vector<float>& T, const std::vector<float>& P);

extern bool itsUseVirtualTemperature;
extern level itsBottomLevel;

}  // namespace si_cuda
}  // namespace plugin
}  // namespace himan

#endif /* HAVE_CUDA */
