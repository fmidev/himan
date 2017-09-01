/**
 * @file metutil.h
 *
 * @brief Meteorological functions' namespace.
 *
 * Namespace tries to combine both CPU and GPU implementations for the same
 * functions.
 */

#ifndef METUTIL_H_
#define METUTIL_H_

#include "assert.h"
#include "cuda_helper.h"
#include "himan_common.h"
#include "numerical_functions.h"

#if defined FAST_MATH and not defined __CUDACC__
#include "fastmath.h"
#define EXP(V) fasterexp(static_cast<float>(V))
#define EXP10(V) fasterexp((static_cast<float>(V)) * 2.30258509299405f)
#define LOG(V) fasterlog(static_cast<float>(V))

inline double fastpow(double a, double b)
{
	// calculate approximation with fraction of the exponent
	int e = (int)b;
	union {
		double d;
		int x[2];
	} u = {a};
	u.x[1] = (int)((b - e) * (u.x[1] - 1072632447) + 1072632447);
	u.x[0] = 0;

	// exponentiation by squaring with the exponent's integer part
	// double r = u.d makes everything much slower, not sure why
	double r = 1.0;
	while (e)
	{
		if (e & 1)
		{
			r *= a;
		}
		a *= a;
		e >>= 1;
	}

	return r * u.d;
}

#define POW(V, E) fastpow(V, E)

#else
#define EXP(V) exp(V)
#define EXP10(V) exp10(V)
#define LOG(V) log(V)
#define POW(V, E) pow(V, E)
#endif

// struct to store LCL level parameters
struct lcl_t
{
	double T;
	double P;
	double Q;

	CUDA_DEVICE
	lcl_t() : T(himan::MissingDouble()), P(himan::MissingDouble()), Q(himan::MissingDouble()) {}
	CUDA_DEVICE
	lcl_t(double T, double P, double Q) : T(T), P(P), Q(Q) {}
};

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

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
 * The caller can the defined which function he wants to call.
 *
 */

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

CUDA_DEVICE
double E_(double R, double P);

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

CUDA_DEVICE
double Es_(double T);

/**
 * @brief Calculates pseudo-adiabatic lapse rate
 *
 * Original author AK Sarkanen May 1985.
 *
 * @param P Pressure in Pa
 * @param T Temperature in K
 * @return Lapse rate in K/km
 */

CUDA_DEVICE
double Gammas_(double P, double T);

/**
 * @brief Calculate moist-adiabatic lapse rate (MALR).
 *
 * Also known as saturated adiabatic lapse rate (SALR)
 *
 * Formula used is (3.16) from
 *
 * Rogers&Yun: A short course in cloud physics 3rd edition
 *
 * combined with gamma (g/Cp) and transformed from m to Pa
 * with inverse of hydropstatic equation.
 *
 * @param P Pressure in Pa
 * @param T Temperature in K
 * @return Lapse rate in K/Pa
 */

CUDA_DEVICE
double Gammaw_(double P, double T);

CUDA_KERNEL
void Gammaw(cdarr_t P, cdarr_t T, darr_t result, size_t N);

/**
 * @brief Calculates the temperature, pressure and specific humidity (Q) of
 * a parcel of air in LCL by vertically iterating from starting height
 *
 * Original author AK Sarkanen/Kalle Eerola
 *
 * @param P Pressure in Pa
 * @param T Temperature in K
 * @param TD Dew point temperature in K
 * @return Pressure (Pa), temperature (K) and specific humidity (g/kg) for LCL .
 */

CUDA_DEVICE
lcl_t LCL_(double P, double T, double TD);

/**
 * @brief LCL level temperature and pressure approximation.
 *
 * For temperature, the used formula is (15) of
 *
 * Bolton: The Computation of Equivalent Potential Temperature (1980)
 *
 * LCL pressure is calculated using starting T and P and T_LCL with
 * Poissons formula assuming dry-adiabatic conditions.
 *
 * @param P Ground pressure in Pa
 * @param T Ground temperature in Kelvin
 * @param TD Ground dewpoint temperature in TD
 * @param Pressure (Pa) and temperature (K) for LCL.
 */
CUDA_DEVICE
lcl_t LCLA_(double P, double T, double TD);

/**
 * @brief Calculate water probability based on T and RH
 *
 * So-called "Koistinen formula"
 *
 * https://wiki.fmi.fi/download/attachments/21139101/IL_olomuototuote_JK.ppt
 *
 * Currently only CPU implementation exists.
 *
 * @param T Surface temperature in K
 * @param RH Surface relative humidity in %
 * @return Water probability
 */

double WaterProbability_(double T, double RH);

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

CUDA_KERNEL
void MixingRatio(cdarr_t T, cdarr_t P, darr_t result, size_t N);

CUDA_DEVICE
double MixingRatio_(double T, double P);

/**
 * @brief Lift a parcel of air to wanted pressure
 *
 * Overcoat for DryLift/MoistLift
 *
 * @param P Initial pressure in Pascals
 * @param T Initial temperature in Kelvins
 * @param TD Initial dewpoint temperature in Kelvins
 * @param targetP Target pressure (where parcel is lifted) in Pascals
 * @return Parcel temperature in wanted pressure in Kelvins
 */

CUDA_KERNEL
void Lift(cdarr_t P, cdarr_t T, cdarr_t TD, cdarr_t targetP, darr_t result, size_t N);

CUDA_DEVICE
double Lift_(double P, double T, double TD, double targetP);

/**
 * @brief Lift a parcel of air to wanted pressure
 *
 * Overcoat for DryLift/MoistLift, with user-given LCL level pressure
 *
 * @param P Initial pressure in Pascals
 * @param T Initial temperature in Kelvins
 * @param PLCL LCL level pressure in Pascals
 * @param targetP Target pressure (where parcel is lifted) in Pascals
 * @return Parcel temperature in wanted pressure in Kelvins
 */

CUDA_KERNEL
void LiftLCL(cdarr_t P, cdarr_t T, cdarr_t LCP, cdarr_t targetP, darr_t result, size_t N);

CUDA_DEVICE
double LiftLCL_(double P, double T, double LCLP, double targetP);

/**
 * @brief Lift a parcel of air moist-adiabatically to wanted pressure
 *
 * Initial temperature is assumed to be saturated ie. LCL level temperature.
 *
 * @param P Pressure of LCL in Pascals
 * @param T Temperature of LCL in K
 * @param targetP Target pressure (where parcel is lifted) in Pascals
 * @return Parcel temperature in wanted pressure in Kelvins
 */

CUDA_KERNEL
void MoistLift(cdarr_t P, cdarr_t T, cdarr_t targetP, darr_t result, size_t N);

CUDA_DEVICE
double MoistLift_(double P, double T, double targetP);

/**
 * @brief Approximate the moist adiabatic lift of an parcel to wanted pressure.
 *
 * Initial temperature is assumed to be saturated.
 * Method used is the (in)famous Wobus function.
 *
 * More information about it can be googled with term 'Wobus function', for example:
 * http://www.caps.ou.edu/ARPS/arpsbrowser/arps5.2.4browser/html_code/adas/mthermo.f90.html
 *
 * Method is faster than the iterative approach but slightly inaccurate
 * (errors up to ~1K, see docs for cape-plugin for more information).
 *
 * @param P Pressure of LCL in Pascals
 * @param T Temperature of LCL in K
 * @param targetP Target pressure (where parcel is lifted) in Pascals
 * @return Parcel temperature in wanted pressure in Kelvins
 */

CUDA_KERNEL
void MoistLiftA(cdarr_t P, cdarr_t T, cdarr_t targetP, darr_t result, size_t N);

CUDA_DEVICE
double MoistLiftA_(double P, double T, double targetP);

/**
 * @brief Lift a parcel of air dry adiabatically to wanted pressure
 *
 * Poissons equation is used
 *
 * @param P Initial pressure in Pascals
 * @param T Initial temperature in Kelvins
 * @param targetP Target pressure (where parcel is lifted) in Pascals
 * @return Parcel temperature in wanted pressure in Kelvins
 */

CUDA_KERNEL
void DryLift(cdarr_t P, cdarr_t T, cdarr_t targetP, darr_t result, size_t N);

CUDA_DEVICE
double DryLift_(double P, double T, double targetP);

/**
 * @brief Calculate dew point temperature from air temperature and relative humidity.
 *
 * The formula used is simple and efficient and gives a close-enough result when humidity
 * is high enough (> 50%).
 *
 * Source: http://journals.ametsoc.org/doi/pdf/10.1175/BAMS-86-2-225
 *
 * @param T Air temperature in Kelvins
 * @param TH Relative humidity in percent
 * @return Dew point temperature in Kelvin
 */

CUDA_DEVICE
double DewPointFromRHSimple_(double T, double RH);

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

CUDA_DEVICE
double DewPointFromRH_(double T, double RH);

/**
 * @brief Calculates Relative Topography between the two given fields in Geop
 *
 *  Currently only CPU implementation exists.

 * @param level1 Value of pressure level1 in Pa
 * @param level2 Value of pressure level2 in Pa
 * @param z1 Geopotential height of level1, Use pressure if level1 = 1000
 * @param z2 Geopotential height of level2
 * @return Relative Topography in Geopotential
 */

double RelativeTopography_(int level1, int level2, double z1, double z2);

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

int LowConvection_(double T0m, double T850);

/**
 * @brief Showalter Index
 *
 * Will lift a parcel of air from 850 hPa to 500 hPa either dry or wet
 * adiabatically.
 *
 * http://forecast.weather.gov/glossary.php?word=SHOWALTER+INDEX
 *
 * @param T850 Temperature of 850 hPa isobar in Kelvins
 * @param T500 Temperature of 500 hPa isobar in Kelvins
 * @param TD850 Dewpoint temperature of 850 hPa isobar in Kelvins
 * @return Index value
 */

CUDA_DEVICE
double SI_(double T850, double T500, double TD850);

/**
 * @brief Cross Totals Index
 *
 * http://glossary.ametsoc.org/wiki/Stability_index
 *
 * @param T500 Temperature of 500 hPa isobar in Kelvins
 * @param TD850 Dewpoint temperature of 850 hPa isobar in Kelvins
 * @return Index value (TD850 - T500)
 */

CUDA_DEVICE
double CTI_(double T500, double TD850);

/**
 * @brief Vertical Totals Index
 *
 * http://glossary.ametsoc.org/wiki/Stability_index
 *
 * @param T850 Temperature of 850 hPa isobar in Kelvins
 * @param T500 Temperature of 500 hPa isobar in Kelvins
 * @return Index value (T850 - T500)
 */

CUDA_DEVICE
double VTI_(double T850, double T500);

/**
 * @brief Total Totals Index
 *
 * http://glossary.ametsoc.org/wiki/Stability_index
 *
 * @param T850 Temperature of 850 hPa isobar in Kelvins
 * @param T500 Temperature of 500 hPa isobar in Kelvins
 * @param TD850 Dewpoint temperature of 850 hPa isobar in Kelvins
 * @return Index value ( T850 - T500 ) + ( TD850 - T500 )
 */

CUDA_DEVICE
double TTI_(double T850, double T500, double TD850);

/**
 * @brief Lifted index
 *
 * http://en.wikipedia.org/wiki/Lifted_index
 *
 * @param T500 Temperature of 500 hPa isobar in Kelvins
 * @param T500m Temperature at 500m above ground in Kelvins
 * @param TD500m Dewpoint temperature at 500m above ground in Kelvins
 * @param P500m Pressure at 500m above ground in Pascals
 * @return Index value
 */

CUDA_DEVICE
double LI_(double T500, double T500m, double TD500m, double P500m);

/**
 * @brief K-Index
 *
 * @param T500 Temperature of 500 hPa isobar in Kelvins
 * @param T700 Temperature of 700 hPa isobar in Kelvins
 * @param T850 Temperature of 850 hPa isobar in Kelvins
 * @param TD700 Dewpoint temperature of 700 hPa isobar in Kelvins
 * @param TD850 Dewpoint temperature of 850 hPa isobar in Kelvins
 * @return Index value
 */

CUDA_DEVICE
double KI_(double T500, double T700, double T850, double TD700, double TD850);

/**
 * @brief Calculate bulk wind shear between two layers in the atmosphere
 *
 * @param U U(upper) - U(lower)
 * @param V V(upper) - V(lower)
 * @return Bulk wind shear in knots
 */

CUDA_DEVICE
double BulkShear_(double U, double V);

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

CUDA_DEVICE
double Tw_(double thetaE, double P);

CUDA_KERNEL
void Tw(cdarr_t thetaE, cdarr_t P, darr_t result, size_t N);

/**
 * @brief Calculate "dry" potential temperature with poissons equation.
 *
 * http://san.hufs.ac.kr/~gwlee/session3/potential.html
 *
 * @param T Temperature in K
 * @param P Pressure in Pa
 * @return Potential temperature in K
 */

CUDA_DEVICE
double Theta_(double T, double P);

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

CUDA_DEVICE
double ThetaE_(double T, double TD, double P);

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

CUDA_DEVICE
double ThetaW_(double thetaE);

/**
 * @brief Calculate virtual temperature
 *
 * Formula is 1.13 from
 *
 * Stull: Meteorology for Scientists and Engineers, 2nd edition (2000)
 *
 * @param T Temperature in K
 * @param P Pressure in Pa
 * @return Virtual temperature in K
 */

CUDA_DEVICE
double VirtualTemperature_(double T, double P);

/**
 *  @brief Calculate the flight level corresponding to given pressure
 *
 *  @param P Pressure in Pa
 *  @return flight level in hecto feet
 */

double FlightLevel_(double P);

#ifdef __CUDACC__

// We have to declare cuda functions in the header or be ready to face the
// eternal horror of 'separate compilation.'

__global__ void DryLift(cdarr_t d_p, cdarr_t d_t, cdarr_t d_targetP, darr_t d_result, size_t N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		d_result[idx] = DryLift_(d_p[idx], d_t[idx], d_targetP[idx]);
	}
}

__global__ void Lift(cdarr_t d_p, cdarr_t d_t, cdarr_t d_td, darr_t d_result, double targetP, size_t N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		d_result[idx] = Lift_(d_p[idx], d_t[idx], d_td[idx], targetP);
	}
}

__global__ void LCL(cdarr_t d_p, cdarr_t d_t, cdarr_t d_td, darr_t d_t_result, darr_t d_p_result, darr_t d_q_result,
                    size_t N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		lcl_t LCL = LCL_(d_p[idx], d_t[idx], d_td[idx]);

		if (d_t_result)
		{
			d_t_result[idx] = LCL.T;
		}

		if (d_p_result)
		{
			d_p_result[idx] = LCL.P;
		}

		if (d_q_result)
		{
			d_q_result[idx] = LCL.Q;
		}
	}
}

#endif

namespace smarttool
{
// smarttool namespace contains functions copied from smarttools with just the most necessary modifications
// made (like using SI units)

// these functions are needed in some applications, but generally they should not be used since their origins
// are unknown

/**
 * @brief Calculate equivalent potential temperature.
 *
 * Original:
 * double NFmiSoundingFunctions::CalcThetaE(double T, double Td, double P)
 *
 * Function has been modified so that it takes humidity as an argument;
 * the original function took dewpoint and calculated humidity from that.
 */

CUDA_DEVICE
double ThetaE_(double T, double RH, double P);

/**
 * Original:
 * double NFmiSoundingFunctions::CalcEs2(double Tcelsius)
 */

CUDA_DEVICE
double Es2_(double T);

/**
 * @brief Mixing ratio formula from smarttool library
 *
 * Original:
 * double NFmiSoundingFunctions::CalcMixingRatio(double T, double Td, double P)
 *
 * Function has been modified so that it takes humidity as an argument;
 * the original function took dewpoint and calculated humidity from that.
 */

CUDA_DEVICE
double MixingRatio_(double T, double RH, double P);

/**
 * Original:
 * double NFmiSoundingFunctions::CalcW(double e, double P)
 */

CUDA_DEVICE
double W_(double e, double P);

CUDA_DEVICE
double E_(double RH, double es);

}  // namespace smarttool
}  // namespace metutil
}  // namespace himan

CUDA_DEVICE
inline double himan::metutil::DewPointFromRHSimple_(double T, double RH) { return (T - ((100 - RH) * 0.2)); }
CUDA_DEVICE
inline double himan::metutil::DewPointFromRH_(double T, double RH)
{
	if (RH == 0.) RH = 0.01;  // formula does not work if RH = 0; actually all small values give extreme Td values
	assert(RH > 0.);
	// assert(RH < 101.);
	assert(T > 0. && T < 500.);

	return (T / (1 - (T * LOG(RH * 0.01) * constants::kRw_div_L)));
}

CUDA_DEVICE
inline double himan::metutil::MixingRatio_(double T, double P)
{
	// Sanity checks
	assert(P > 1000);
	assert(T > 0 && T < 500);

	double E = Es_(T);  // Pa

	return 621.97 * E / (P - E);
}

CUDA_DEVICE
inline double himan::metutil::E_(double R, double P)
{
	assert(P > 1000);
	assert(R > 0.001);

	// R is g/kg, converting it to g/g gives multiplier 1000

	return (R * P / (constants::kEp * 1000));
}

CUDA_DEVICE
inline double himan::metutil::DryLift_(double P, double T, double targetP)
{
	if (targetP >= P)
	{
		return MissingDouble();
	}

	// Sanity checks
	assert(P > 10000);
	assert(T > 100 && T < 400);
	assert(targetP > 10000);

	return T * pow((targetP / P), 0.286);
}

CUDA_DEVICE
inline double himan::metutil::Lift_(double P, double T, double TD, double targetP)
{
	// Search LCL level
	lcl_t LCL = metutil::LCLA_(P, T, TD);

	if (LCL.P < targetP)
	{
		// LCL level is higher than requested pressure, only dry lift is needed
		return DryLift_(P, T, targetP);
	}

	return MoistLift_(P, T, targetP);
}

CUDA_DEVICE
inline double himan::metutil::LiftLCL_(double P, double T, double LCLP, double targetP)
{
	if (LCLP < targetP)
	{
		// LCL level is higher than requested pressure, only dry lift is needed
		return DryLift_(P, T, targetP);
	}

	// Wanted height is above LCL
	if (P < LCLP)
	{
		// Current level is above LCL, only moist lift is required
		return MoistLift_(P, T, targetP);
	}

	// First lift dry adiabatically to LCL height
	double LCLT = DryLift_(P, T, LCLP);

	// Lift from LCL to wanted pressure
	return MoistLift_(LCLP, LCLT, targetP);
}

CUDA_DEVICE
inline double himan::metutil::MoistLift_(double P, double T, double targetP)
{
	if (IsMissingDouble(T) || IsMissingDouble(P) || targetP >= P)
	{
		return MissingDouble();
	}

	// Sanity checks

	assert(P > 2000);
	assert(T > 100 && T < 400);
	assert(targetP > 2000);

	double Pint = P;  // Pa
	double Tint = T;  // K

	/*
	 * Units: Temperature in Kelvins, Pressure in Pascals
	 */

	double T0 = Tint;

	int i = 0;
	const double Pstep = 100;  // Pa; do not increase this as quality of results is weakened
	const int maxIter = static_cast<int>(100000 / Pstep + 10);  // varadutuaan iteroimaan 1000hPa --> 0 hPa + marginaali

	double value = MissingDouble();

	while (++i < maxIter)
	{
		Tint = T0 - metutil::Gammaw_(Pint, Tint) * Pstep;

		assert(Tint == Tint);

		Pint -= Pstep;

		if (Pint <= targetP)
		{
			value = himan::numerical_functions::interpolation::Linear(targetP, Pint, Pint + Pstep, T0, Tint);
			break;
		}

		T0 = Tint;
	}

	return value;
}

CUDA_DEVICE
inline double Wobf(double T)
{
	// "Wobus function" is a polynomial approximation of moist lift
	// process. It is called from MoistLiftA_().

	double ret = himan::MissingDouble();

	T -= 20;

	if (T <= 0)
	{
		ret = 1 +
		      T * (-8.841660499999999e-3 +
		           T * (1.4714143e-4 + T * (-9.671989000000001e-7 + T * (-3.2607217e-8 + T * (-3.8598073e-10)))));
		ret = 15.130 / (ret * ret * ret * ret);
	}
	else
	{
		ret = 1 +
		      T * (3.6182989e-03 +
		           T * (-1.3603273e-05 +
		                T * (4.9618922e-07 +
		                     T * (-6.1059365e-09 + T * (3.9401551e-11 + T * (-1.2588129e-13 + T * (1.6688280e-16)))))));
		ret = (29.930 / (ret * ret * ret * ret)) + (0.96 * T) - 14.8;
	}

	return ret;
}

CUDA_DEVICE
inline double himan::metutil::MoistLiftA_(double P, double T, double targetP)
{
	if (IsMissingDouble(T) || IsMissingDouble(P) || targetP >= P)
	{
		return MissingDouble();
	}

	using namespace himan::constants;

	const double theta = Theta_(T, P) - kKelvin;  // pot temp, C
	T -= kKelvin;

	const double thetaw = theta - Wobf(theta) + Wobf(T);  // moist pot temp, C

	double remains = 9999;  // try to minimize this
	double ratio = 1;

	const double pwrp = POW(targetP / 100000, kRd_div_Cp);  // exner

	double t1 = (thetaw + kKelvin) * pwrp - kKelvin;
	double e1 = Wobf(t1) - Wobf(thetaw);
	double t2 = t1 - (e1 * ratio);                 // improved estimate of return value (saturated lifted temperature)
	double pot = (t2 + kKelvin) / pwrp - kKelvin;  // pot temperature of t2 at pressure p
	double e2 = pot + Wobf(t2) - Wobf(pot) - thetaw;

	while (fabs(remains) - 0.1 > 0)
	{
		ratio = (t2 - t1) / (e2 - e1);

		t1 = t2;
		e1 = e2;
		t2 = t1 - e1 * ratio;
		pot = (t2 + kKelvin) / pwrp - kKelvin;
		e2 = pot + Wobf(t2) - Wobf(pot) - thetaw;

		remains = e2 * ratio;
	}

	return t2 - remains + kKelvin;
}

CUDA_DEVICE
inline lcl_t himan::metutil::LCL_(double P, double T, double TD)
{
	// Sanity checks

	assert(P > 10000);
	assert(T > 0);
	assert(T < 500);
	assert(TD > 0);
	assert(TD < 500);

	// starting T step

	double Tstep = 0.05;

	P *= 0.01;  // HPa

	// saturated vapor pressure

	double E0 = himan::metutil::Es_(TD) * 0.01;  // HPa

	double Q = constants::kEp * E0 / P;
	double C = T / pow(E0, constants::kRd_div_Cp);

	double TLCL = MissingDouble();
	double PLCL = MissingDouble();

	double Torig = T;
	double Porig = P;

	short nq = 0;

	lcl_t ret;

	while (++nq < 100)
	{
		double TEs = C * pow(himan::metutil::Es_(T) * 0.01, constants::kRd_div_Cp);

		if (fabs(TEs - T) < 0.05)
		{
			TLCL = T;
			PLCL = pow((TLCL / Torig), (1 / constants::kRd_div_Cp)) * P;

			ret.P = PLCL * 100;  // Pa

			ret.T = TLCL;  // K

			ret.Q = Q;
		}
		else
		{
			Tstep = MIN((TEs - T) / (2 * (nq + 1)), 15.);
			T -= Tstep;
		}
	}

	// Fallback to slower method

	if (IsMissingDouble(ret.P))
	{
		T = Torig;
		Tstep = 0.1;

		nq = 0;

		while (++nq <= 500)
		{
			if ((C * pow(himan::metutil::Es_(T) * 0.01, constants::kRd_div_Cp) - T) > 0)
			{
				T -= Tstep;
			}
			else
			{
				TLCL = T;
				PLCL = pow(TLCL / Torig, (1 / constants::kRd_div_Cp)) * Porig;

				ret.P = PLCL * 100;  // Pa

				ret.T = TLCL;  // K

				ret.Q = Q;

				break;
			}
		}
	}

	return ret;
}

CUDA_DEVICE
inline lcl_t himan::metutil::LCLA_(double P, double T, double TD)
{
	lcl_t ret;

	// Sanity checks

	assert(P > 10000);
	assert(T > 0);
	assert(T < 500);
	assert(TD > 0 && TD != 56);
	assert(TD < 500);

	double A = 1 / (TD - 56);
	double B = log(T / TD) / 800.;

	ret.T = 1 / (A + B) + 56;
	ret.P = P * pow((ret.T / T), 3.5011);

	return ret;
}

CUDA_DEVICE
inline double himan::metutil::Es_(double T)
{
	// Sanity checks
	assert(T == T && T > 0 && T < 500);  // check also NaN

	double Es;

	T -= himan::constants::kKelvin;

	if (T > -5)
	{
		Es = 6.107 * EXP10(7.5 * T / (237.0 + T));
	}
	else
	{
		Es = 6.107 * EXP10(9.5 * T / (265.5 + T));
	}

	assert(Es == Es);  // check NaN

	return 100 * Es;  // Pa
}

CUDA_DEVICE
inline double himan::metutil::Gammas_(double P, double T)
{
	// Sanity checks

	assert(P > 10000);
	assert(T > 0 && T < 500);

	// http://glossary.ametsoc.org/wiki/Pseudoadiabatic_lapse_rate

	// specific humidity: http://glossary.ametsoc.org/wiki/Specific_humidity

	namespace hc = himan::constants;

	double Q = hc::kEp * (himan::metutil::Es_(T) * 0.01) / (P * 0.01);

	double A = hc::kRd * T / hc::kCp / P * (1 + hc::kL * Q / hc::kRd / T);

	return A / (1 + hc::kEp / hc::kCp * (hc::kL * hc::kL) / hc::kRd * Q / (T * T));
}

CUDA_DEVICE
inline double himan::metutil::Gammaw_(double P, double T)
{
	// Sanity checks

	assert(P > 1000);
	assert(T > 0 && T < 500);

	namespace hc = himan::constants;

	double esat = Es_(T);
	double wsat = hc::kEp * esat / (P - esat);  // Rogers&Yun 2.18
	double numerator = (2. / 7.) * T + (2. / 7. * hc::kL / hc::kRd) * wsat;
	double denominator = P * (1 + (hc::kEp * hc::kL * hc::kL / (hc::kRd * hc::kCp)) * wsat / (T * T));

	assert(numerator != 0);
	assert(denominator != 0);

	return numerator / denominator;  // Rogers&Yun 3.16

	/*	double r = himan::metutil::MixingRatio_(T, P);

	    double numerator = hc::kG * (1 + (hc::kL * r) / (hc::kRd * T));
	    double denominator = hc::kCp + ((hc::kL*hc::kL * r * hc::kEp) / (hc::kRd * T * T));

	    return numerator / denominator;
	 * */
}

CUDA_DEVICE
inline double himan::metutil::CTI_(double TD850, double T500) { return TD850 - T500; }
CUDA_DEVICE
inline double himan::metutil::VTI_(double T850, double T500) { return T850 - T500; }
CUDA_DEVICE
inline double himan::metutil::TTI_(double T850, double T500, double TD850)
{
	return CTI_(TD850, T500) + VTI_(T850, T500);
}

CUDA_DEVICE
inline double himan::metutil::KI_(double T850, double T700, double T500, double TD850, double TD700)
{
	return (T850 - T500 + TD850 - (T700 - TD700)) - constants::kKelvin;
}

CUDA_DEVICE
inline double himan::metutil::LI_(double T500, double T500m, double TD500m, double P500m)
{
	lcl_t LCL = LCL_(50000, T500m, TD500m);

	double li = MissingDouble();

	const double TARGET_PRESSURE = 50000;

/*	if (IsMissingDouble(LCL.P))
	{
		return li;
	}
*/
	if (LCL.P <= 85000)
	{
		// LCL pressure is below wanted pressure, no need to do wet-adiabatic
		// lifting

		double dryT = DryLift_(P500m, T500m, TARGET_PRESSURE);

		li = T500 - dryT;
	}
	else if (LCL.P > 85000)
	{
		// Grid point is inside or above cloud

		double wetT = Lift_(P500m, T500m, TD500m, TARGET_PRESSURE);

		li = T500 - wetT;
	}

	return li;
}

CUDA_DEVICE
inline double himan::metutil::SI_(double T850, double T500, double TD850)
{
	lcl_t LCL = metutil::LCL_(85000, T850, TD850);

	double si = MissingDouble();

	const double TARGET_PRESSURE = 50000;

/*	if (IsMissingDouble(LCL.P))
	{
		return si;
	}
*/
	if (LCL.P <= 85000)
	{
		// LCL pressure is below wanted pressure, no need to do wet-adiabatic
		// lifting

		double dryT = DryLift_(85000, T850, TARGET_PRESSURE);

		si = T500 - dryT;
	}
	else if (LCL.P > 85000)
	{
		// Grid point is inside or above cloud

		double wetT = Lift_(85000, T850, TD850, TARGET_PRESSURE);

		si = T500 - wetT;
	}

	return si;
}

CUDA_DEVICE
inline double himan::metutil::BulkShear_(double U, double V)
{
	return sqrt(U * U + V * V) * 1.943844492;  // converting to knots
}

CUDA_DEVICE
inline double himan::metutil::Theta_(double T, double P)
{
	assert(T > 0);
	assert(P > 1000);

	return T * pow((100000. / P), 0.28586);
}

CUDA_DEVICE
inline double himan::metutil::ThetaE_(double T, double TD, double P)
{
	assert(T > 0);
	assert(P > 1000);

	// Get LCL temperature
	const double A = 1 / (TD - 56);
	const double B = log(T / TD) / 800.;
	const double TLCL = 1 / (A + B) + 56;

	// Mixing ratio at initial level
	const double r = himan::metutil::MixingRatio_(T, P);

	// 100000 = reference pressure 1000hPa
	const double C = T * pow(100000. / P, 0.2854 * (1 - 0.00028 * r));
	const double D = 3.376 / TLCL - 0.00254;
	const double F = r * (1 + 0.00081 * r);

	return C * EXP(D * F);
}

CUDA_DEVICE
inline double himan::metutil::Tw_(double thetaE, double P)
{
	assert(thetaE > 0);
	assert(P > 1000);

	if (IsMissingDouble(thetaE) || IsMissingDouble(P)) return MissingDouble();

	using namespace himan::constants;

	const double a = 17.67;
	const double b = 243.5;  // K
	const double P0 = 100000;
	const double lambda = 1 / kRd_div_Cp;
	const double C = kKelvin;
	const double pi = pow(P / P0, kRd_div_Cp);  // Nondimensional pressure, exner function

	const double Te = thetaE * pi;  // Equivalent temperature
	const double ratio = pow((C / Te), lambda);

	// Quadratic regression curves for thetaW

	const double k1 = -38.5 * pi * pi + 137.81 * pi - 53.737;
	const double k2 = -4.392 * pi * pi + 56.831 * pi - 0.384;

	const double p0 = P0 * 0.01;
	const double p = P * 0.01;

	// Regression line for transition points of different Tw formulas

	const double Dp = 1 / (0.1859 * p / p0 + 0.6512);

	double Tw = MissingDouble();

	if (ratio > Dp)
	{
		const double A = 2675;  // K

		// e & r as Davies-Jones implemented them
		const double e = 6.112 * exp((a * (Te - C)) / (Te - C + b));
		const double r = kEp * e / (p0 * pow(pi, lambda) - e);

		const double nomin = A * r;
		const double denom = 1 + A * r * ((a * b) / pow((Te - C + b), 2));

		Tw = Te - C - nomin / denom;
	}
	else
	{
		const double hot = (Te > 355.15) ? 1 : 0;
		const double cold = (ratio >= 1 && ratio <= Dp) ? 0 : 1;

		Tw = k1 - 1.21 * cold - 1.45 * hot - (k2 - 1.21 * cold) * ratio + (0.58 / ratio) * hot;
	}

	Tw += C;

	// Improve accuracy with Newton-Raphson
	// Current Tw is used as initial guess

	double remains = 1e38;
	const int maxiter = 5;
	int iter = 0;

	while (remains > 0.01 && iter < maxiter)
	{
		const double newRatio = pow((C / Tw), lambda);

		const double e = 6.112 * exp((a * (Tw - C)) / (Tw - C + b));
		const double r = kEp * e / (p0 * pow(pi, lambda) - e);

		// Evaluate f(x)

		const double A = 1 - e / (p0 * pow(pi, lambda));
		const double B = (3036 / Tw - 1.78) * (r + 0.448 * r * r);

		const double fTw = newRatio * pow(A, kRd_div_Cp * lambda) * exp(-lambda * B);

		// Partial derivative de/dTw
		const double deTw = e * (a * b) / pow((Tw - C + b), 2);

		// Partial derivative dr/dTw
		const double drTw = ((kEp * p) / pow((p - e), 2)) * deTw;

		// Evaluate f'(x)

		const double A_ = (1 / Tw) + (kRd_div_Cp / (p - e)) * deTw;

		const double B_ = -3036 * (r + 0.448 * r * r) / (Tw * Tw);
		const double C_ = (3036 / Tw - 1.78) * (1 + 2 * (0.448 * r)) * drTw;

		const double dTw = -lambda * (A_ + B_ + C_);

		double newTw = Tw - (fTw - ratio) / dTw;

		iter++;
		remains = fabs(newTw - Tw);
		Tw = newTw;
	}

	return Tw;
}

CUDA_DEVICE
inline double himan::metutil::ThetaW_(double thetaE)
{
	double thetaW = thetaE;

	if (thetaE >= 173.15)
	{
		const double X = thetaE / constants::kKelvin;

		const double a0 = 7.101574;
		const double a1 = -20.68208;
		const double a2 = 16.11182;
		const double a3 = 2.574631;
		const double a4 = -5.205688;
		const double b1 = -3.552497;
		const double b2 = 3.781782;
		const double b3 = -0.6899655;
		const double b4 = -0.5929340;

		const double A = a0 + a1 * X + a2 * X * X + a3 * pow(X, 3.) + a4 * pow(X, 4.);
		const double B = 1 + b1 * X + b2 * X * X + b3 * pow(X, 3.) + b4 * pow(X, 4.);

		thetaW = thetaW - exp(A / B);
	}

	return thetaW;
}

CUDA_DEVICE
inline double himan::metutil::VirtualTemperature_(double T, double P)
{
	assert(T > 100);
	assert(T < 400);
	assert(P > 1000);

	double r = 0.001 * MixingRatio_(T, P);  // kg/kg
	return (1 + 0.61 * r) * T;
}

CUDA_DEVICE
inline double himan::metutil::smarttool::Es2_(double T)
{
	assert(T > 100);
	assert(T < 350);

	const double b = 17.2694;
	const double e0 = 6.11;   // 6.11 <- 0.611 [kPa]
	const double T2 = 35.86;  // [K]

	double nume = b * (T - himan::constants::kKelvin);
	double deno = (T - T2);

	return e0 * ::exp(nume / deno);
}

CUDA_DEVICE
inline double himan::metutil::smarttool::E_(double RH, double es)
{
	assert(RH >= 0);
	assert(RH < 102);

	return RH * es / 100;
}

CUDA_DEVICE
inline double himan::metutil::smarttool::ThetaE_(double T, double RH, double P)
{
	assert(RH >= 0);
	assert(RH < 102);
	assert(T > 150);
	assert(T < 350);
	assert(P > 1500);

	double tpot = himan::metutil::Theta_(T, P);
	double w = himan::metutil::smarttool::MixingRatio_(T, RH, P);
	return tpot + 3 * w;
}

CUDA_DEVICE
inline double himan::metutil::smarttool::W_(double e, double P)
{
	assert(P > 1500);

	double w = 0.622 * e / P * 100000;
	assert(w < 60);

	return w;
}

CUDA_DEVICE
inline double himan::metutil::smarttool::MixingRatio_(double T, double RH, double P)
{
	assert(RH >= 0);
	assert(RH < 102);
	assert(T > 150);
	assert(T < 350);
	assert(P > 1500);

	double es = himan::metutil::smarttool::Es2_(T);
	double e = himan::metutil::smarttool::E_(RH, es);
	double w = himan::metutil::smarttool::W_(e, P);

	return w;
}

#endif /* METUTIL_H_ */
