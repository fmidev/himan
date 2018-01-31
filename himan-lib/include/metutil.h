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

template <typename A>
int LowConvection_(A T0m, A T850)
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

}  // namespace metutil
}  // namespace himan
