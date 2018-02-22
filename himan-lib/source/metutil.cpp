/**
 * @file metutil.cpp
 *
 * @brief Different utility functions in a namespace
 *
 */

#include "metutil.h"

using namespace himan;

double metutil::FlightLevel_(double P)
{
	// Return missing value if missing value is passed as an argument. This is done because missing value was turned
	// into -nan in some unknown way, probably inside a math function.
	if (IsMissing(P))
		return P;

	// International Standard Atmosphere Conditions

	// average temperature lapse rate within the troposphere
	const double gamma = -0.0065;  // [K/m]
	// ground temperature
	const double TG = 288.15;  // [K]
	// ground pressure
	const double PG = 101325;  // [Pa]
	// tropopause altitude
	const double h_tropo = 11000;  //[m]
	// tropopause pressure
	const double p_tropo = 22632;  // [Pa]
	// tropopause temperature
	const double T_tropo = 216.65;  // [K]

	// conversion m->hft
	const double m_hft = 0.032808399;

	double h;

	// troposphere
	if (P > p_tropo)
	{
		h = (std::pow(P / PG, -(constants::kRd * gamma) / constants::kG) - 1) * TG / gamma * m_hft;
	}
	// above tropopause
	else
	{
		h = (-std::log(P / p_tropo) * constants::kRd * T_tropo / constants::kG + h_tropo) * m_hft;
	}

	// round to multiple of 5
	return std::round(h / 5.) * 5.;
}
