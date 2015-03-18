/**
 * @file metutil.h
 *
 * @date Apr 29, 2014
 * @author partio
 *
 * @brief Meteorological functions' namespace.
 *
 * Namespace tries to combine both CPU and GPU implementations for the same
 * functions.
 */

#ifndef METUTIL_H_
#define METUTIL_H_

#include "assert.h"
#include "himan_common.h"
//#include "cuda_helper.h"

#ifdef __CUDACC__
#define CUDA_DEVICE __device__
#define CUDA_KERNEL __global__
#else
#define CUDA_DEVICE
#define CUDA_KERNEL
#endif

// struct to store LCL level parameters
struct lcl_t
{
	double T;
	double P;
	double Q;

	CUDA_DEVICE
	lcl_t()
	: T(himan::kFloatMissing)
	, P(himan::kFloatMissing)
	, Q(himan::kFloatMissing)
	{}
};

#define MIN(a,b) (((a)<(b))?(a):(b))

typedef double* __restrict darr_t;
typedef const double* __restrict cdarr_t;

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
 * @brief Calculate water vapor saturated pressure in Pa
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
 * http://en.wikipedia.org/wiki/Lapse_rate#Saturated_adiabatic_lapse_rate
 *
 * @param P Pressure in Pa
 * @param T Temperature in K
 * @return Lapse rate in K/km
 */

CUDA_DEVICE
double Gammaw_(double P, double T);

/**
 * @brief Calculates the temperature, pressure and specific humidity (Q) of
 * a parcel of air in LCL
 *
 * Original author AK Sarkanen/Kalle Eerola
 *
 * @param P Pressure in Pa
 * @param T Temperature in K
 * @param TD Dew point temperature in K
 * @return Pressure (Pa), temperature (K) and specific humidity (g/kg) for LCL (in this order).
 */

CUDA_DEVICE
lcl_t LCL_(double P, double T, double TD);

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
 * @brief Lift a parcel of air moist-adiabatically to wanted pressure
 *
 * Function will calculate LCL from given arguments and starts
 * lifting from that pressure and temperature.
 *
 * @param P Initial pressure in Pascals
 * @param T Initial temperature in Kelvins
 * @param TD Initial dewpoint temperature in Kelvins
 * @param targetP Target pressure (where parcel is lifted) in Pascals
 * @return Parcel temperature in wanted pressure in Kelvins
 */

CUDA_KERNEL
void MoistLift(cdarr_t P, cdarr_t T, cdarr_t TD, darr_t result, double targetP, size_t N);

CUDA_DEVICE
double MoistLift_(double P, double T, double TD, double targetP);

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
void DryLift(cdarr_t P, cdarr_t T, darr_t result, double targetP, size_t N);

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

#ifdef __CUDACC__

// We have to declare cuda functions in the header or be ready to face the
// eternal horror of 'separate compilation.'

__global__ void DryLift(cdarr_t d_p, cdarr_t d_t, darr_t d_result, double targetP, size_t N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		d_result[idx] = DryLift_(d_p[idx], d_t[idx], targetP);
	}
}

__global__ void MoistLift(cdarr_t d_p, cdarr_t d_t, cdarr_t d_td, darr_t d_result, double targetP, size_t N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		d_result[idx] = MoistLift_(d_p[idx], d_t[idx], d_td[idx], targetP);
	}
}

__global__ void LCL(cdarr_t d_p, cdarr_t d_t, cdarr_t d_td, darr_t d_t_result, darr_t d_p_result, darr_t d_q_result, size_t N)
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

} // namespace metutil
} // namespace himan

CUDA_DEVICE
inline double himan::metutil::DewPointFromRHSimple_(double T, double RH)
{
	return (T - ((100 - RH) * 0.2));
}

CUDA_DEVICE
inline double himan::metutil::DewPointFromRH_(double T, double RH)
{
	return (T / (1 - (T * log(RH * 0.01) * constants::kRw_div_L)));
}

CUDA_DEVICE
inline double himan::metutil::MixingRatio_(double T, double P)
{
	// Sanity checks
	assert(P > 1000);
	assert(T > 0 && T < 500);

	double E = Es_(T) * 0.01; // hPa

	P *= 0.01;

	return 621.97 * E / (P - E);
}

CUDA_DEVICE
inline double himan::metutil::DryLift_(double P, double T, double targetP)
{
	// Sanity checks
	assert(P > 1000);
	assert(T > 0 && T < 500);
	assert(targetP > 10000);

	return T * pow((targetP / P), 0.286);
}

CUDA_DEVICE
inline double himan::metutil::MoistLift_(double P, double T, double TD, double targetP)
{
	// Sanity checks
	assert(P > 10000);
	assert(T > 0 && T < 500);
	assert(TD > 0 && TD < 500);
	assert(targetP > 10000);

	// Search LCL level
	lcl_t LCL = metutil::LCL_(P, T, TD);

	double Pint = LCL.P; // Pa
	double Tint = LCL.T; // K

	// Start moist lifting from LCL height

	double value = kFloatMissing;

	if (Tint == kFloatMissing || Pint == kFloatMissing)
	{
		return kFloatMissing;
	}
	else
	{
		/*
		 * Units: Temperature in Kelvins, Pressure in Pascals
		 */

		double T0 = Tint;

		//double Z = kFloatMissing;

		int i = 0;
		const double Pstep = 100; // Pa

		while (++i < 500) // usually we don't reach this value
		{
			double TA = Tint;
/*
			if (i <= 2)
			{
				Z = i * Pstep/2;
			}
			else
			{
				Z = 2 * Pstep;
			}
*/
			// Gammaw() takes Pa
			Tint = T0 - metutil::Gammaw_(Pint, Tint) * Pstep;

			if (i > 2)
			{
				T0 = TA;
			}

			Pint -= Pstep;

			if (Pint <= targetP)
			{
				value = Tint;
				break;
			}
		}
	}

	return value;
}

CUDA_DEVICE
inline
lcl_t himan::metutil::LCL_(double P, double T, double TD)
{
	// Sanity checks

	assert(P > 10000);
	assert(T > 0);
	assert(T < 500);
	assert(TD > 0);
	assert(TD < 500);

	// starting T step

	double Tstep = 0.05;

	const double kRCp = 0.286;

	P *= 0.01; // HPa

	// saturated vapor pressure

	double E0 = himan::metutil::Es_(TD) * 0.01; // HPa

	double Q = constants::kEp * E0 / P;
	double C = T / pow(E0, kRCp);

	double TLCL = kFloatMissing;
	double PLCL = kFloatMissing;

	double Torig = T;
	double Porig = P;

	short nq = 0;

	lcl_t ret;
	
	while (++nq < 100)
	{
		double TEs = C * pow(himan::metutil::Es_(T)*0.01, kRCp);

		if (fabs(TEs - T) < 0.05)
		{
			TLCL = T;
			PLCL = pow((TLCL/Torig), (1/kRCp)) * P;

			ret.P = PLCL * 100; // Pa
			ret.T = (TLCL == kFloatMissing) ? kFloatMissing : TLCL; // K
			ret.Q = Q;

		}
		else
		{
			Tstep = MIN((TEs - T) / (2 * (nq+1)), 15.);
			T -= Tstep;
		}
	}

	// Fallback to slower method

	if (ret.P == kFloatMissing)
	{
		T = Torig;
		Tstep = 0.1;

		nq = 0;

		while (++nq <= 500)
		{
			if ((C * pow(himan::metutil::Es_(T)*0.01, kRCp)-T) > 0)
			{
				T -= Tstep;
			}
			else
			{
				TLCL = T;
				PLCL = pow(TLCL / Torig, (1/kRCp)) * Porig;

				ret.P = PLCL * 100; // Pa
				ret.T = (TLCL == kFloatMissing) ? kFloatMissing : TLCL; // K
				ret.Q = Q;

				break;
			}
		}
	}
	
	return ret;
}

CUDA_DEVICE
inline
double himan::metutil::Es_(double T)
{
	// Sanity checks
	assert(T == T && T > 0 && T < 500); // check also NaN

	double Es;

	if (T == kFloatMissing)
	{
		return kFloatMissing;
	}

	T -= himan::constants::kKelvin;

	if (T > -5)
	{
		Es = 6.107 * exp10(7.5*T/(237.0+T));
	}
	else
	{
		Es = 6.107 * exp10(9.5*T/(265.5+T));
	}
	
	assert(Es == Es); // check NaN

	return 100 * Es; // Pa

}

CUDA_DEVICE
inline
double himan::metutil::Gammas_(double P, double T)
{
	// Sanity checks

	assert(P > 10000);
	assert(T > 0 && T < 500);

	// http://glossary.ametsoc.org/wiki/Pseudoadiabatic_lapse_rate

	// specific humidity: http://glossary.ametsoc.org/wiki/Specific_humidity

	namespace hc = himan::constants;

	double Q = hc::kEp * (himan::metutil::Es_(T) * 0.01) / (P*0.01);

	double A = hc::kRd * T / hc::kCp / P * (1+hc::kL*Q/hc::kRd/T);

	return A / (1 + hc::kEp / hc::kCp * (hc::kL * hc::kL) / hc::kRd * Q / (T*T));
}

CUDA_DEVICE
inline
double himan::metutil::Gammaw_(double P, double T)
{
	// Sanity checks

	assert(P > 10000);
	assert(T > 0 && T < 500);

	/*
	 * Constants:
	 * - g = 9.81
	 * - Cp = 1003.5
	 * - L = 2.5e6
	 *
	 * Variables:
	 * - dWs = saturation mixing ratio = util::MixingRatio()
	 */

	namespace hc = himan::constants;

	double r = himan::metutil::MixingRatio_(T, P);
	const double kL = 2.256e6; // Another kL !!!

	double numerator = hc::kG * (1 + (kL * r) / (hc::kRd * T));
	double denominator = hc::kCp + ((kL*kL * r * hc::kEp) / (hc::kRd * T * T));

	return numerator / denominator;
}

CUDA_DEVICE
inline
double himan::metutil::CTI_(double TD850, double T500)
{
	return TD850 - T500;
}

CUDA_DEVICE
inline
double himan::metutil::VTI_(double T850, double T500)
{
	return T850 - T500;
}

CUDA_DEVICE
inline
double himan::metutil::TTI_(double T850, double T500, double TD850)
{
	return CTI_(TD850, T500) + VTI_(T850, T500);
}

CUDA_DEVICE
inline
double himan::metutil::KI_(double T850, double T700, double T500, double TD850, double TD700)
{
	return (T850 - T500 + TD850 - (T700 - TD700)) - constants::kKelvin;
}

CUDA_DEVICE
inline
double himan::metutil::LI_(double T500, double T500m, double TD500m, double P500m)
{
	lcl_t LCL = LCL_(50000, T500m, TD500m);

	double li = kFloatMissing;

	const double TARGET_PRESSURE = 50000;

	if (LCL.P == kFloatMissing)
	{
		return li;
	}

	if (LCL.P <= 85000)
	{
		// LCL pressure is below wanted pressure, no need to do wet-adiabatic
		// lifting

		double dryT = DryLift_(P500m, T500m, TARGET_PRESSURE);

		if (dryT != kFloatMissing)
		{
			li = T500 - dryT;
		}
	}
	else
	{
		// Grid point is inside or above cloud

		double wetT = MoistLift_(P500m, T500m, TD500m, TARGET_PRESSURE);

		if (wetT != kFloatMissing)
		{
			li = T500 - wetT;
		}
	}

	return li;
}

CUDA_DEVICE
inline
double himan::metutil::SI_(double T850, double T500, double TD850)
{
	lcl_t LCL = metutil::LCL_(85000, T850, TD850);

	double si = kFloatMissing;

	const double TARGET_PRESSURE = 50000;

	if (LCL.P == kFloatMissing)
	{
		return si;
	}
	
	if (LCL.P <= 85000)
	{
		// LCL pressure is below wanted pressure, no need to do wet-adiabatic
		// lifting

		double dryT = DryLift_(85000, T850, TARGET_PRESSURE);
		
		if (dryT != kFloatMissing)
		{
			si = T500 - dryT;
		}
	}
	else
	{
		// Grid point is inside or above cloud
		
		double wetT = MoistLift_(85000, T850, TD850, TARGET_PRESSURE);

		if (wetT != kFloatMissing)
		{
			si = T500 - wetT;
		}
	}

	return si;
}

CUDA_DEVICE
inline
double himan::metutil::BulkShear_(double U, double V)
{
	return sqrt(U*U + V*V) * 1.943844492; // converting to knots
}

#endif /* METUTIL_H_ */
