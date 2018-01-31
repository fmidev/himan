#pragma once

#include "cuda_helper.h"
#include "himan_common.h"

namespace himan
{
namespace metutil
{
/**
 * @brief Calculate partial water vapor pressure from mixing ratio and pressure
 *
 * Basically this is inverse of mixing ratio, formula used is (2.18)
 * from Rogers & Yau: A Short Course in Cloud Physics (3rd edition).
 *
 * @param R Mixing ratio in g/kg
 * @param P Pressure in Pa
 * @return Water vapour pressure in Pa
 */

template <typename A>
CUDA_DEVICE A E_(A R, A P)
{
	ASSERT(P > 1000);
	ASSERT(R > 0.001);

	// R is g/kg, converting it to g/g gives multiplier 1000

	return (R * P / (himan::constants::kEp * 1000));
}

/**
 * @brief Calculate water vapor saturated pressure in Pa
 *
 * Equation inherited from hilake.
 *
 * Equation found in f.ex. Smithsonian meteorological tables or
 * http://www.srh.noaa.gov/images/epz/wxcalc/vaporPressure.pdf
 *
 * If temperature is less than -5, use ice instead of water for
 * calculations.
 *
 * If T is actually TD, actual water vapor pressure is calculated
 *
 * @param T Temperature in K
 * @return (Saturated) water vapor pressure in Pa
 */

template <typename A>
CUDA_DEVICE A Es_(A T)
{
	ASSERT((T == T && T > 0 && T < 500) || IsMissing(T));

	A Es;

	T -= himan::constants::kKelvin;

	if (T > -5)
	{
		Es = 6.107 * exp10(7.5 * T / (237.0 + T));
	}
	else
	{
		Es = 6.107 * exp10(9.5 * T / (265.5 + T));
	}

	return 100 * Es;  // Pa
}

/**
 * @brief Calculate actual or saturated mixing ratio.
 *
 * If first argument is temperature, saturated mixing ratio is calculated.
 * If first argument is dewpoint temperature, actual mixing ratio is calculated.
 *
 * http://www.srh.noaa.gov/images/epz/wxcalc/mixingRatio.pdf
 *
 * @param T Temperature or dewpoint temperature in K
 * @param P Pressure in Pa
 * @return Mixing ration in g/kg
 */

template <typename A>
CUDA_DEVICE A MixingRatio_(A T, A P)
{
	ASSERT(P > 1000 || IsMissing(T));
	ASSERT((T > 0 && T < 500) || IsMissing(T));

	const A E = Es_<A>(T);  // Pa

	return 621.97 * E / (P - E);
}

/**
 * @brief Calculate dew point temperature from air temperature and relative humidity.
 *
 * The formula used is a more complex one than in DewPointSimple_, but gives more accurate
 * results when humidity is low.
 *
 * Source: http://journals.ametsoc.org/doi/pdf/10.1175/BAMS-86-2-225
 *
 * @param T Air temperature in Kelvins
 * @param TH Relative humidity in percent
 * @return Dew point temperature in Kelvin
 */

template <typename A>
CUDA_DEVICE A DewPointFromRH_(A T, A RH)
{
	if (RH == 0.)
	{
		RH = 0.01;  // formula does not work if RH = 0; actually all small values give extreme Td values
	}

	ASSERT((RH > 0. && RH < 101.) || IsMissing(RH));
	ASSERT((T > 0. && T < 500.) || IsMissing(T));

	return (T / (1 - (T * log(RH * 0.01) * himan::constants::kRw_div_L)));
}

/**
 * @brief Calculate adiabatic saturation temperature at given pressure.
 *
 * Function approximates the temperature at given pressure when equivalent
 * potential temperature is given. ThetaE should be evaluated at saturation (LCL)
 * level.
 *
 * Maximum relative error in the initial guess compared to integrated value is
 * 1.8K; the value is further corrected with Newton-Raphson so that the error
 * reduces to 0.34K.
 *
 * Formula derived by Davies-Jones in:
 *
 * An Efficient and Accurate Method for Computing the Wet-Bulb Temperature
 * along Pseudoadiabat (2007)
 *
 * @param thetaE Equivalent potential temperature in K
 * @param P Target pressure in Pa
 * @return Temperature along a saturated adiabat at wanted height in Kelvins
 */

template <typename A>
CUDA_DEVICE A Tw_(A thetaE, A P)
{
	ASSERT(thetaE > 0);
	ASSERT(P > 1000);

	if (IsMissing(thetaE) || IsMissing(P))
	{
		return himan::MissingValue<A>();
	}

	using namespace himan::constants;

	const A a = 17.67;
	const A b = 243.5;  // K
	const A P0 = 100000;
	const A lambda = 1 / kRd_div_Cp;
	const A C = kKelvin;
	const A pi = pow(P / P0, kRd_div_Cp);  // Nondimensional pressure, exner function

	const A Te = thetaE * pi;  // Equivalent temperature
	const A ratio = pow((C / Te), lambda);

	// Quadratic regression curves for thetaW

	const A k1 = -38.5 * pi * pi + 137.81 * pi - 53.737;
	const A k2 = -4.392 * pi * pi + 56.831 * pi - 0.384;

	const A p0 = P0 * 0.01;
	const A p = P * 0.01;

	// Regression line for transition points of different Tw formulas

	const A Dp = 1 / (0.1859 * p / p0 + 0.6512);

	A Tw;

	if (ratio > Dp)
	{
		const A A_ = 2675;  // K

		// e & r as Davies-Jones implemented them
		const A e = 6.112 * exp((a * (Te - C)) / (Te - C + b));
		const A r = kEp * e / (p0 * pow(pi, lambda) - e);

		const A nomin = A_ * r;
		const A denom = 1 + A_ * r * ((a * b) / pow((Te - C + b), 2));

		Tw = Te - C - nomin / denom;
	}
	else
	{
		const A hot = (Te > 355.15) ? 1 : 0;
		const A cold = (ratio >= 1 && ratio <= Dp) ? 0 : 1;

		Tw = k1 - 1.21 * cold - 1.45 * hot - (k2 - 1.21 * cold) * ratio + (0.58 / ratio) * hot;
	}

	Tw += C;

	// Improve accuracy with Newton-Raphson
	// Current Tw is used as initial guess

	A remains = 1e38;
	const int maxiter = 5;
	int iter = 0;

	while (remains > 0.01 && iter < maxiter)
	{
		const A newRatio = pow((C / Tw), lambda);

		const A e = 6.112 * exp((a * (Tw - C)) / (Tw - C + b));
		const A r = kEp * e / (p0 * pow(pi, lambda) - e);

		// Evaluate f(x)

		const A A_ = 1 - e / (p0 * pow(pi, lambda));
		const A B = (3036 / Tw - 1.78) * (r + 0.448 * r * r);

		const A fTw = newRatio * pow(A_, kRd_div_Cp * lambda) * exp(-lambda * B);

		// Partial derivative de/dTw
		const A deTw = e * (a * b) / pow((Tw - C + b), 2);

		// Partial derivative dr/dTw
		const A drTw = ((kEp * p) / pow((p - e), 2)) * deTw;

		// Evaluate f'(x)

		const A A__ = (1 / Tw) + (kRd_div_Cp / (p - e)) * deTw;

		const A B_ = -3036 * (r + 0.448 * r * r) / (Tw * Tw);
		const A C_ = (3036 / Tw - 1.78) * (1 + 2 * (0.448 * r)) * drTw;

		const A dTw = -lambda * (A__ + B_ + C_);

		const A newTw = Tw - (fTw - ratio) / dTw;

		iter++;
		remains = fabs(newTw - Tw);
		Tw = newTw;
	}

	return Tw;
}

/**
 * @brief Calculate "dry" potential temperature with poissons equation.
 *
 * http://san.hufs.ac.kr/~gwlee/session3/potential.html
 *
 * @param T Temperature in K
 * @param P Pressure in Pa
 * @return Potential temperature in K
 */

template <typename A>
CUDA_DEVICE A Theta_(A T, A P)
{
	ASSERT(T > 0 || IsMissing(T));
	ASSERT(P > 1000);

	return T * pow((100000. / P), 0.28586);
}

/**
 * @brief Calculate equivalent potential temperature.
 *
 * Formula used is (43) from
 *
 * Bolton: The Computation of Equivalent Potential Temperature (1980)
*
 * @param T Temperature at initial level (Kelvin)
 * @param TD Dewpoint temperature at inital level (Kelvin)
 * @param P Pressure at initial level (Pa)
 * @return Equivalent potential temperature ThetaE in Kelvins
 */

template <typename A>
CUDA_DEVICE A ThetaE_(A T, A TD, A P)
{
	ASSERT(T > 0 || IsMissing(T));
	ASSERT(P > 1000);

	// Get LCL temperature
	const A A_ = 1 / (TD - 56);
	const A B_ = log(T / TD) / 800.;
	const A TLCL = 1 / (A_ + B_) + 56;

	// Mixing ratio at initial level
	const A r = himan::metutil::MixingRatio_<A>(T, P);

	// 100000 = reference pressure 1000hPa
	const A C = T * pow(100000. / P, 0.2854 * (1 - 0.00028 * r));
	const A D = 3.376 / TLCL - 0.00254;
	const A F = r * (1 + 0.00081 * r);

	return C * exp(D * F);
}

/**
 * @brief Calculate wet-bulb potential temperature
 *
 * Formula used is (3.8) from
 *
 * Davies-Jones: An Efficient and Accurate Method for Computing the Wet-Bulb Temperature
 * along Pseudoadiabats (2007)
 *
 * @param thetaE Equivalent potential temperature, Kelvin
 * @return Wet-bulb potential temperature ThetaW in Kelvins
 */

template <typename A>
CUDA_DEVICE A ThetaW_(A thetaE)
{
	A thetaW = thetaE;

	if (thetaE >= 173.15)
	{
		const A X = thetaE / constants::kKelvin;

		const A a0 = 7.101574;
		const A a1 = -20.68208;
		const A a2 = 16.11182;
		const A a3 = 2.574631;
		const A a4 = -5.205688;
		const A b1 = -3.552497;
		const A b2 = 3.781782;
		const A b3 = -0.6899655;
		const A b4 = -0.5929340;

		const A A_ = a0 + a1 * X + a2 * X * X + a3 * pow(X, 3.) + a4 * pow(X, 4.);
		const A B_ = 1 + b1 * X + b2 * X * X + b3 * pow(X, 3.) + b4 * pow(X, 4.);

		thetaW = thetaW - exp(A_ / B_);
	}

	return thetaW;
}

template <typename A>
CUDA_DEVICE A VirtualTemperature_(A T, A P)
{
	ASSERT(IsMissing(T) || T > 100 || T < 400);
	ASSERT(IsMissing(P) || P > 1000);

	A r = 0.001 * MixingRatio_<A>(T, P);  // kg/kg
	return (1 + 0.61 * r) * T;
}

// smarttool namespace contains functions copied from smarttools with just the most necessary modifications
// made (like using SI units)

// these functions are needed in some applications, but generally they should not be used since their origins
// are unknown

namespace smarttool
{
/**
 * Original:
 * double NFmiSoundingFunctions::CalcEs2(double Tcelsius)
 */

template <typename A>
CUDA_DEVICE A Es2_(A T)
{
	ASSERT(T > 100 || IsMissing(T));
	ASSERT(T < 350 || IsMissing(T));

	const A b = 17.2694;
	const A e0 = 6.11;   // 6.11 <- 0.611 [kPa]
	const A T2 = 35.86;  // [K]

	const A nume = b * (T - himan::constants::kKelvin);
	const A deno = (T - T2);

	return e0 * ::exp(nume / deno);
}

/**
 * Original:
 * double NFmiSoundingFunctions::CalcW(double e, double P)
 */

template <typename A>
CUDA_DEVICE double W_(A e, A P)
{
	ASSERT(P > 1500);

	const A w = 0.622 * e / P * 100000.;
	ASSERT(w < 60);

	return w;
}

template <typename A>
CUDA_DEVICE A E_(A RH, A es)
{
	ASSERT(RH >= 0);
	ASSERT(RH < 102);

	return RH * es / 100.;
}

/**
 * @brief Mixing ratio formula from smarttool library
 *
 * Original:
 * double NFmiSoundingFunctions::CalcMixingRatio(double T, double Td, double P)
 *
 * Function has been modified so that it takes humidity as an argument;
 * the original function took dewpoint and calculated humidity from that.
 */

template <typename A>
CUDA_DEVICE A MixingRatio_(A T, A RH, A P)
{
	ASSERT(RH >= 0);
	ASSERT(RH < 102);
	ASSERT(T > 150 || IsMissing(T));
	ASSERT(T < 350 || IsMissing(T));
	ASSERT(P > 1500);

	const A es = himan::metutil::smarttool::Es2_(T);
	const A e = himan::metutil::smarttool::E_(RH, es);
	const A w = himan::metutil::smarttool::W_(e, P);

	return w;
}

}  // namespace smarttool
}  // namespace metutil
}  // namespace himan
