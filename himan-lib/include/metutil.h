#pragma once

#include "cuda_helper.h"
#include "himan_common.h"
#include "numerical_functions.h"

namespace himan
{
/**
 * @namespace himan::metutil
 * @brief Namespace for all meteorological utility functions
 *
 * The naming convention is as follows:
 * - If a function is calculating something over whole array, it is named without any
 *   underscores. For example DryLift(). This function type is just an overcoat for
 *   the actual function which performs the calculation.
 * - If a function is doing the actual calculation, and underscore is appended to function
 *   name. For example, DryLift_().
 *
 */

namespace metutil
{
/**
 * @brief Checks the stability of surface and 850hPa level
 * 0 = no convection
 * 1 = low convection during on sea
 * 2 = low convection during summer on land
 * T0m height was commented as 2m, but calculated as 0m temperature in hil_pp
 *
 * Currently only CPU implementation exists.
 *
 * @param T0m Value of 0m temperature in K
 * @param T850 Value of temperature at 850 hPa pressure level in K
 * @return convection value.
 */

template <typename Type>
int LowConvection_(Type T0m, Type T850)
{
	ASSERT(T0m > 0);
	ASSERT(T850 > 0);

	T0m -= constants::kKelvin;
	T850 -= constants::kKelvin;

	// Lability during summer (T0m > 8C)
	if (T0m >= 8 && T0m - T850 >= 10)
	{
		return 2;
	}

	// Lability during winter (T850 < 0C) Probably above sea
	else if (T0m >= 0 && T850 <= 0 && T0m - T850 >= 10)
	{
		return 1;
	}

	return 0;
}

/**
 *  @brief Calculate the flight level corresponding to given pressure
 *
 *  @param P Pressure in Pa
 *  @return flight level in hecto feet
 */

double FlightLevel_(double P);

/**
 * @brief Calculate (local) solar time
 *
 * https://www.pveducation.org/pvcdrom/properties-of-sunlight/solar-time
 *
 * Equation of Time correction is not used.
 *
 * @param latlon point in question in latitude longitude coordinates
 * @param rt time in question in UTC time zone
 * @return local solar time
 */

raw_time SolarTime_(const point& latlon, const raw_time& rt);

/**
 * @brief Calculate elevation angle.
 *
 * Function taken from newbase, comments from there:
 * * [Venäläinen: lisenssiaattitutkimus ANNEX (A3)]
 * * These calculations of a elevation angle of the Sun does not give very
 * * accurate estimations. This inaccuracy is partly caused by
 * * the inaccurate calculations of declination angle.
 * * The time given as a parameter must be a local solar time.
 *
 * * [Venäläinen: lisenssiaattitutkimus ANNEX (A4)]
 * * The time given as a parameter must be a local solar time.
 * * These calculations of a declination angle of the Sun does not give very accurate
 * * estimations. May be the inaccuracy is mostly caused because of inaccurate
 * * calculation of the yearAngle (a measure to Earths circulation around the Sun).
 * * This inaccuracy causes an inaccurate elevation angle. But the calculation
 * * of the elevation angle uses also the method NFmiLocation::AtzimuthAngle(),
 * * where calculates a correction term, which quite successfully compensates
 * * errors of the elevation angle.
 *
 * * The time given as a parameter must be a local solar time. Here the
 * * atzimuth angle = 0 when the Sun is in the North.
 * * There's also an odd correction term anEqualizerTerm (Laura Thölix:
 * * pro gradu, page 16), which perhaps somehow compensates the inaccurate
 * * calculation of the yearAngle (a measure to the Earths circulaton around the Sun).
 *
 */

double ElevationAngle_(const point& latlon, const raw_time& rt);

}  // namespace metutil
}  // namespace himan
