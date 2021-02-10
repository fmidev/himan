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

raw_time metutil::SolarTime_(const point& latlon, const raw_time& rt)
{
	// const double B = (360. / 365) * (stoi(rt.String("%j")) - 81);
	// const double EoT = 9.87 * sin(2 * B) - 7.53 * cos(B) - 1.5 * sin(B);
	const double EoT = 0;
	const double TC = 4 * latlon.X() + EoT;
	return rt + time_duration("00:00:01") * static_cast<int>(TC) * 60;
}

double metutil::ElevationAngle_(const point& latlon, const raw_time& rt)
{
	const raw_time solarTime = SolarTime_(latlon, rt);
	const double julianDay = stod(solarTime.String("%j"));

	const double hour =
	    (stod(solarTime.String("%H")) + stod(solarTime.String("%M")) / 60. + stod(solarTime.String("%S")) / 3600.);
	const double yearAngle = 2.0 * M_PI * (julianDay + (hour / 24.) - 1.0) / 365.;

	double siny, siny2, siny3, cosy, cosy2, cosy3;
	sincos(yearAngle, &siny, &cosy);
	sincos(yearAngle * 2, &siny2, &cosy2);
	sincos(yearAngle * 3, &siny3, &cosy3);

	const double decAngle = 0.006918 - 0.399912 * cosy + 0.070257 * siny - 0.006758 * cosy2 + 0.000907 * siny2 -
	                        0.002697 * cosy3 + 0.00148 * siny3;

	const double anEqualizerTerm = (0.0172 + 0.4281 * cosy - 7.3515 * siny - 3.3495 * cosy2 - 9.3619 * siny2) / 60.;

	const double azimAngle = (hour + anEqualizerTerm) * M_PI / 12.;

	double sinl, cosl, sind, cosd;
	sincos(latlon.Y() * constants::kDeg, &sinl, &cosl);
	sincos(decAngle, &sind, &cosd);

	const double elevAngle = asin(sind * sinl - cosd * cosl * cos(azimAngle));
	return constants::kRad * elevAngle;
}
