/**
 * @file   si_cuda.h
 *
 */

#pragma once

#include "himan_common.h"
#include "info.h"
#include "metutil.h"
#include "numerical_functions.h"

/*
 * Namespace CAPE holds common CAPE integration functions
 * that are called from CPU and GPU code.
 */

#define LINEAR himan::numerical_functions::interpolation::Linear

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
inline double IntegrateEnteringParcel(double Tenv, double prevTenv, double Tparcel, double prevTparcel, double Zenv,
                                      double prevZenv)
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

	if (!(intersection.Y() == intersection.Y())) return 0;

	prevZenv = intersection.Y();

	double value = himan::constants::kG * (Zenv - prevZenv) * ((Tparcel - Tenv) / Tenv);
	value = fmin(150, fmax(-150., value));

	assert(!isnan(value) && !isinf(value));

	return value;
}

CUDA_DEVICE
inline void IntegrateLeavingParcel(double Tenv, double prevTenv, double Tparcel, double prevTparcel, double Penv,
                                   double prevPenv, double Zenv, double prevZenv, double& out_value, double& out_ELT,
                                   double& out_ELP)
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

	out_ELT = himan::MissingDouble();
	out_ELP = himan::MissingDouble();

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

	Zenv = intersectionZ.Y();
	assert(fabs(intersectionZ.X() - intersectionP.X()) < 1.);
	double value = himan::constants::kG * (Zenv - prevZenv) * ((prevTparcel - prevTenv) / prevTenv);
	value = fmin(150, fmax(-150., value));

	assert(!isnan(value) && !isinf(value));

	out_value = value;
	out_ELT = intersectionP.X();
	out_ELP = intersectionP.Y();
}

CUDA_DEVICE
inline double IntegrateHeightAreaLeavingParcel(double Tenv, double prevTenv, double Tparcel, double prevTparcel,
                                               double Zenv, double prevZenv, double areaUpperLimit)
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

	double newTenv = LINEAR(areaUpperLimit, Zenv, prevZenv, Tenv, prevTenv);
	double newTparcel = LINEAR(areaUpperLimit, Zenv, prevZenv, Tparcel, prevTparcel);

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

	assert(newTparcel >= newTenv);
	assert(areaUpperLimit > prevZenv);

	double CAPE = himan::constants::kG * (areaUpperLimit - prevZenv) * ((prevTparcel - prevTenv) / prevTenv);

	CAPE = fmin(CAPE, 150.);

	assert(CAPE >= 0.);
	assert(CAPE <= 150);

	return CAPE;
}

CUDA_DEVICE
inline double IntegrateTemperatureAreaEnteringParcel(double Tenv, double prevTenv, double Tparcel, double prevTparcel,
                                                     double Zenv, double prevZenv, double areaColderLimit,
                                                     double areaWarmerLimit)
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

	double areaLimit;
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

	double newPrevZenv = LINEAR(areaLimit, Tenv, prevTenv, Zenv, prevZenv);
	double newTparcel = LINEAR(newPrevZenv, Zenv, prevZenv, Tparcel, prevTparcel);

	if (newTparcel < areaLimit)
	{
		// Tparcel has to be warmer than environment, otherwise no CAPE

		for (int i = 0; i < 20; i++)
		{
			areaLimit += (fromWarmerToCold) ? -0.1 : 0.1;

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

	assert(Tparcel >= Tenv);
	assert(Zenv >= newPrevZenv);

	double CAPE = himan::constants::kG * (Zenv - newPrevZenv) * ((Tparcel - Tenv) / Tenv);
	CAPE = fmin(CAPE, 150.);

	assert(Zenv >= prevZenv);
	assert(CAPE >= 0.);
	assert(CAPE <= 150.);

	return CAPE;
}

CUDA_DEVICE
inline double IntegrateTemperatureAreaLeavingParcel(double Tenv, double prevTenv, double Tparcel, double prevTparcel,
                                                    double Zenv, double prevZenv, double areaColderLimit,
                                                    double areaWarmerLimit)
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

	double areaLimit;
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

	double newZenv = LINEAR(areaLimit, Tenv, prevTenv, Zenv, prevZenv);
	double newTparcel = LINEAR(newZenv, Zenv, prevZenv, Tparcel, prevTparcel);

	if (newTparcel <= areaLimit)
	{
		// Tparcel has to be warmer than environment, otherwise no CAPE

		for (int i = 0; i < 20; i++)
		{
			areaLimit += (fromColdToWarmer) ? -0.1 : 0.1;

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

	assert(Tparcel >= Tenv);
	assert(newZenv <= Zenv);
	assert(newZenv >= prevZenv);

	double CAPE = himan::constants::kG * (Zenv - prevZenv) * ((newTparcel - areaLimit) / areaLimit);
	assert(CAPE >= 0.);

	CAPE = fmin(CAPE, 150.);

	assert(CAPE <= 150.);

	return CAPE;
}

CUDA_DEVICE
inline double CalcCAPE1040(double Tenv, double prevTenv, double Tparcel, double prevTparcel, double Penv,
                           double prevPenv, double Zenv, double prevZenv)
{
	double C = 0;

	assert((Tenv == Tenv) && (Penv == Penv) && (Tparcel == Tparcel));

	if (Tparcel < Tenv && prevTparcel < prevTenv)
	{
		// No CAPE
		return C;
	}

	Tenv = himan::metutil::VirtualTemperature_(Tenv, Penv * 100);
	Tparcel = himan::metutil::VirtualTemperature_(Tparcel, Penv * 100);

	prevTenv = himan::metutil::VirtualTemperature_(prevTenv, prevPenv * 100);
	prevTparcel = himan::metutil::VirtualTemperature_(prevTparcel, prevPenv * 100);

	double coldColderLimit = 233.15;
	double coldWarmerLimit = 263.15;

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
				C = himan::constants::kG * (Zenv - prevZenv) * ((Tparcel - Tenv) / Tenv);
				assert(C >= 0.);
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
			double CAPE, ELT, ELP;
			CAPE::IntegrateLeavingParcel(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv, CAPE,
			                             ELT, ELP);
			C = CAPE;
		}
	}

	return C;
}

CUDA_DEVICE
inline double CalcCAPE3km(double Tenv, double prevTenv, double Tparcel, double prevTparcel, double Penv,
                          double prevPenv, double Zenv, double prevZenv)
{
	double C = 0.;

	Tenv = himan::metutil::VirtualTemperature_(Tenv, Penv * 100);
	Tparcel = himan::metutil::VirtualTemperature_(Tparcel, Penv * 100);

	prevTenv = himan::metutil::VirtualTemperature_(prevTenv, prevPenv * 100);
	prevTparcel = himan::metutil::VirtualTemperature_(prevTparcel, prevPenv * 100);

	if (Tparcel > Tenv)
	{
		// Have buoyancy at current height

		if (Zenv <= 3000.)
		{
			if (prevTparcel >= prevTenv)
			{
				// Firmly in the zone
				C = himan::constants::kG * (Zenv - prevZenv) * ((Tparcel - Tenv) / Tenv);
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
				double CAPE, ELT, ELP;
				CAPE::IntegrateLeavingParcel(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv, CAPE,
				                             ELT, ELP);
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
inline void CalcCAPE(double Tenv, double prevTenv, double Tparcel, double prevTparcel, double Penv, double prevPenv,
                     double Zenv, double prevZenv, double& out_CAPE, double& out_ELT, double& out_ELP)
{
	out_CAPE = 0.;

	out_ELT = himan::MissingDouble();
	out_ELP = himan::MissingDouble();

	assert((Tenv == Tenv) && (Penv == Penv) && (Tparcel == Tparcel));

	if (Tparcel < Tenv && prevTparcel < prevTenv)
	{
		// No CAPE
		return;
	}

	Tenv = himan::metutil::VirtualTemperature_(Tenv, Penv * 100);
	Tparcel = himan::metutil::VirtualTemperature_(Tparcel, Penv * 100);

	prevTenv = himan::metutil::VirtualTemperature_(prevTenv, prevPenv * 100);
	prevTparcel = himan::metutil::VirtualTemperature_(prevTparcel, prevPenv * 100);

	if (Tparcel >= Tenv && prevTparcel >= Tenv)
	{
		// We are fully in a CAPE zone
		out_CAPE = himan::constants::kG * (Zenv - prevZenv) * ((Tparcel - Tenv) / Tenv);
	}
	else if (Tparcel >= Tenv && prevTparcel < prevTenv)
	{
		out_CAPE = CAPE::IntegrateEnteringParcel(Tenv, prevTenv, Tparcel, prevTparcel, Zenv, prevZenv);
	}
	else if (Tparcel < Tenv && prevTparcel >= prevTenv)
	{
		CAPE::IntegrateLeavingParcel(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv, out_CAPE,
		                             out_ELT, out_ELP);
	}

	assert(out_CAPE >= 0);
}

CUDA_DEVICE
inline double CalcCIN(double Tenv, double prevTenv, double Tparcel, double prevTparcel, double Penv, double prevPenv,
                      double Zenv, double prevZenv)
{
	if (Tparcel >= Tenv && prevTparcel >= prevTenv)
	{
		// No CIN
		return 0;
	}

	double cin = 0;

	if (Tparcel < Tenv && prevTparcel < Tenv)
	{
		// We are fully in a CIN zone
		cin = himan::constants::kG * (Zenv - prevZenv) * ((Tparcel - Tenv) / Tenv);
	}
	else if (Tparcel < Tenv && prevTparcel >= prevTenv)
	{
		cin = CAPE::IntegrateEnteringParcel(Tenv, prevTenv, Tparcel, prevTparcel, Zenv, prevZenv);
	}
	else if (Tparcel >= Tenv && prevTparcel < prevTenv)
	{
		double cin, a, b;
		CAPE::IntegrateLeavingParcel(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv, cin, a, b);
	}

	return cin;
}

}  // namespace CAPE

#ifdef HAVE_CUDA
#include "cuda_helper.h"
#include "info_simple.h"
#include "plugin_configuration.h"

typedef std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> cape_source;

namespace himan
{
namespace plugin
{
namespace cape_cuda
{
cape_source GetHighestThetaEValuesGPU(const std::shared_ptr<const plugin_configuration> conf,
                                      std::shared_ptr<info> myTargetInfo);
std::pair<std::vector<double>, std::vector<double>> GetLFCGPU(const std::shared_ptr<const plugin_configuration> conf,
                                                              std::shared_ptr<info> myTargetInfo,
                                                              std::vector<double>& T, std::vector<double>& P,
                                                              std::vector<double>& TenvLCL);
cape_source Get500mMixingRatioValuesGPU(std::shared_ptr<const plugin_configuration> conf,
                                        std::shared_ptr<info> myTargetInfo);
void GetCINGPU(const std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<info> myTargetInfo,
               const std::vector<double>& Tsource, const std::vector<double>& Psource, const std::vector<double>& TLCL,
               const std::vector<double>& PLCL, const std::vector<double>& PLFC, param CINParam);
void GetCAPEGPU(const std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<info> myTargetInfo,
                const std::vector<double>& T, const std::vector<double>& P, himan::param ELTParam,
                himan::param ELPParam, himan::param CAPEParam, himan::param CAPE1040Param, himan::param CAPE3kmParam);

extern level itsBottomLevel;

}  // namespace si_cuda
}  // namespace plugin
}  // namespace himan

#endif /* HAVE_CUDA */
