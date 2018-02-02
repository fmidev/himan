#pragma once

#include "moisture.h"
#include "numerical_functions.h"

namespace himan
{
namespace metutil
{
// struct to store LCL level parameters
template <typename Type>
struct lcl_t
{
	Type T;
	Type P;
	Type Q;

	CUDA_DEVICE
	lcl_t() : T(MissingValue<Type>()), P(MissingValue<Type>()), Q(MissingValue<Type>())
	{
	}

	CUDA_DEVICE
	lcl_t(Type T, Type P, Type Q) : T(T), P(P), Q(Q)
	{
	}
};

template <typename Type>
CUDA_DEVICE Type Wobf(Type T)
{
	// "Wobus function" is a polynomial approximation of moist lift
	// process. It is called from MoistLiftA_().

	Type ret;

	T -= 20;

	if (T <= 0)
	{
		const Type A = static_cast<Type>(-8.841660499999999e-3);
		const Type B = static_cast<Type>(1.4714143e-4);
		const Type C = static_cast<Type>(-9.671989000000001e-7);
		const Type D = static_cast<Type>(-3.2607217e-8);
		const Type E = static_cast<Type>(-3.8598073e-10);

		ret = 1 + T * (A + T * (B + T * (C + T * (D + T * E))));
		ret = static_cast<Type>(15.13) / (ret * ret * ret * ret);
	}
	else
	{
		const Type A = static_cast<Type>(3.6182989e-03);
		const Type B = static_cast<Type>(-1.3603273e-05);
		const Type C = static_cast<Type>(4.9618922e-07);
		const Type D = static_cast<Type>(-6.1059365e-09);
		const Type E = static_cast<Type>(3.9401551e-11);
		const Type F = static_cast<Type>(-1.2588129e-13);
		const Type G = static_cast<Type>(1.6688280e-16);

		ret = 1 + T * (A + T * (B + T * (C + T * (D + T * (E + T * (F + T * G))))));
		ret = (static_cast<Type>(29.93) / (ret * ret * ret * ret)) + (static_cast<Type>(0.96) * T) -
		      static_cast<Type>(14.8);
	}

	return ret;
}

/**
 * @brief Calculates pseudo-adiabatic lapse rate
 *
 * Original author AK Sarkanen May 1985.
 *
 * @param P Pressure in Pa
 * @param T Temperature in K
 * @return Lapse rate in K/km
 */

template <typename Type>
CUDA_DEVICE Type Gammas_(Type P, Type T)
{
	ASSERT(P > 10000);
	ASSERT((T > 0 && T < 500) || IsMissing(T));

	// http://glossary.ametsoc.org/wiki/Pseudoadiabatic_lapse_rate
	// specific humidity: http://glossary.ametsoc.org/wiki/Specific_humidity

	namespace hc = constants;

	Type Q = hc::kEp * (::himan::metutil::Es_<Type>(T) * 0.01) / (P * 0.01);

	Type B = hc::kRd * T / hc::kCp / P * (1 + hc::kL * Q / hc::kRd / T);

	return B / (1 + hc::kEp / hc::kCp * (hc::kL * hc::kL) / hc::kRd * Q / (T * T));
}
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

template <typename Type>
CUDA_DEVICE Type Gammaw_(Type P, Type T)
{
	// Sanity checks

	ASSERT(P > 1000);
	ASSERT((T > 0 && T < 500) || IsMissing(T));

	namespace hc = constants;

	Type esat = ::himan::metutil::Es_<Type>(T);
	Type wsat = hc::kEp * esat / (P - esat);  // Rogers&Yun 2.18
	Type numerator = (2. / 7.) * T + (2. / 7. * hc::kL / hc::kRd) * wsat;
	Type denominator = P * (1 + (hc::kEp * hc::kL * hc::kL / (hc::kRd * hc::kCp)) * wsat / (T * T));

	ASSERT(numerator != 0);
	ASSERT(denominator != 0);

	return numerator / denominator;  // Rogers&Yun 3.16
}

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

template <typename Type>
CUDA_DEVICE lcl_t<Type> LCL_(Type P, Type T, Type TD)
{
	ASSERT(P > 10000);
	ASSERT(T > 0 || IsMissing(T));
	ASSERT(T < 500 || IsMissing(T));
	ASSERT(TD > 0);
	ASSERT(TD < 500);

	// starting T step

	Type Tstep = 0.05;

	P *= 0.01;  // HPa

	// saturated vapor pressure

	const Type E0 = himan::metutil::Es_<Type>(TD) * 0.01;  // HPa

	const Type Q = constants::kEp * E0 / P;
	const Type C = T / std::pow(E0, constants::kRd_div_Cp);

	Type TLCL, PLCL;

	Type Torig = T;
	Type Porig = P;

	short nq = 0;

	lcl_t<Type> ret;

	while (++nq < 100)
	{
		const Type TEs = C * std::pow(himan::metutil::Es_<Type>(T) * 0.01, constants::kRd_div_Cp);

		if (fabs(TEs - T) < 0.05)
		{
			TLCL = T;
			PLCL = std::pow((TLCL / Torig), (1 / constants::kRd_div_Cp)) * P;

			ret.P = PLCL * 100;  // Pa

			ret.T = TLCL;  // K

			ret.Q = Q;
		}
		else
		{
			Tstep = fmin((TEs - T) / (2 * (nq + 1)), 15.);
			T -= Tstep;
		}
	}

	// Fallback to slower method

	if (IsMissing(ret.P))
	{
		T = Torig;
		Tstep = 0.1;

		nq = 0;

		while (++nq <= 500)
		{
			if ((C * std::pow(himan::metutil::Es_<Type>(T) * 0.01, constants::kRd_div_Cp) - T) > 0)
			{
				T -= Tstep;
			}
			else
			{
				TLCL = T;
				PLCL = std::pow(TLCL / Torig, (1 / constants::kRd_div_Cp)) * Porig;

				ret.P = PLCL * 100;  // Pa

				ret.T = TLCL;  // K

				ret.Q = Q;

				break;
			}
		}
	}

	return ret;
}
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

template <typename Type>
CUDA_DEVICE lcl_t<Type> LCLA_(Type P, Type T, Type TD)
{
	lcl_t<Type> ret;

	// Sanity checks

	ASSERT(P > 10000);
	ASSERT(T > 0 || IsMissing(T));
	ASSERT(T < 500 || IsMissing(T));
	ASSERT(TD > 0 && TD != 56);
	ASSERT(TD < 500);

	Type B = 1 / (TD - 56);
	Type C = log(T / TD) / 800.;

	ret.T = 1 / (B + C) + 56;
	ret.P = P * std::pow((ret.T / T), 3.5011);

	return ret;
}

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

template <typename Type>
CUDA_DEVICE Type DryLift_(Type P, Type T, Type targetP)
{
	if (targetP >= P)
	{
		return MissingValue<Type>();
	}

	// Sanity checks
	ASSERT(IsMissing(P) || P > 10000);
	ASSERT(IsMissing(T) || (T > 100 && T < 400));
	ASSERT(targetP > 10000);

	return T * std::pow((targetP / P), 0.286);
}

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

template <typename Type>
CUDA_DEVICE Type MoistLift_(Type P, Type T, Type targetP)
{
	if (IsMissing(T) || IsMissing(P) || targetP >= P)
	{
		return MissingValue<Type>();
	}

	// Sanity checks

	ASSERT(P > 2000);
	ASSERT((T > 100 && T < 400) || IsMissing(T));
	ASSERT(targetP > 2000);

	Type Pint = P;  // Pa
	Type Tint = T;  // K

	/*
	 * Units: Temperature in Kelvins, Pressure in Pascals
	 */

	Type T0 = Tint;

	int i = 0;
	const Type Pstep = 100;  // Pa; do not increase this as quality of results is weakened
	const int maxIter = static_cast<int>(100000 / Pstep + 10);  // varadutuaan iteroimaan 1000hPa --> 0 hPa + marginaali

	Type value = MissingValue<Type>();

	while (++i < maxIter)
	{
		Tint = T0 - metutil::Gammaw_(Pint, Tint) * Pstep;

		ASSERT(Tint == Tint);

		Pint -= Pstep;

		if (Pint <= targetP)
		{
			value = ::himan::numerical_functions::interpolation::Linear(targetP, Pint, Pint + Pstep, T0, Tint);
			break;
		}

		T0 = Tint;
	}

	return value;
}

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

template <typename Type>
CUDA_DEVICE Type MoistLiftA_(Type P, Type T, Type targetP)
{
	if (IsMissing(T) || IsMissing(P) || targetP >= P)
	{
		return MissingValue<Type>();
	}

	using namespace constants;
	const Type kelvin = static_cast<Type>(kKelvin);

	const Type theta = Theta_(T, P) - kelvin;  // pot temp, C
	T -= kelvin;

	const Type thetaw = theta - Wobf<Type>(theta) + Wobf<Type>(T);  // moist pot temp, C

	Type remains = 9999;  // try to minimize this
	Type ratio = 1;

	const Type pwrp = std::pow(targetP / 100000, static_cast<Type>(kRd_div_Cp));  // exner

	Type t1 = (thetaw + kelvin) * pwrp - kelvin;
	Type e1 = Wobf<Type>(t1) - Wobf<Type>(thetaw);
	Type t2 = t1 - (e1 * ratio);               // improved estimate of return value (saturated lifted temperature)
	Type pot = (t2 + kelvin) / pwrp - kelvin;  // pot temperature of t2 at pressure p
	Type e2 = pot + Wobf<Type>(t2) - Wobf<Type>(pot) - thetaw;

	while (fabs(remains) - 0.1 > 0)
	{
		ratio = (t2 - t1) / (e2 - e1);

		t1 = t2;
		e1 = e2;
		t2 = t1 - e1 * ratio;
		pot = (t2 + kelvin) / pwrp - kelvin;
		e2 = pot + Wobf<Type>(t2) - Wobf<Type>(pot) - thetaw;

		remains = e2 * ratio;
	}

	return t2 - remains + kelvin;
}

/**
 * @brief Lift a parcel of air to wanted pressure
 *
 * Overcoat for DryLift/MoistLift, with user-given LCL level pressure
 *
 * A-version used LCL approximation functions.
 *
 * @param P Initial pressure in Pascals
 * @param T Initial temperature in Kelvins
 * @param PLCL LCL level pressure in Pascals
 * @param targetP Target pressure (where parcel is lifted) in Pascals
 * @return Parcel temperature in wanted pressure in Kelvins
 */

template <typename Type>
CUDA_DEVICE Type LiftLCL_(Type P, Type T, Type LCLP, Type targetP)
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
	const Type LCLT = DryLift_<Type>(P, T, LCLP);

	// Lift from LCL to wanted pressure
	return MoistLift_<Type>(LCLP, LCLT, targetP);
}

template <typename Type>
CUDA_DEVICE double LiftLCLA_(Type P, Type T, Type LCLP, Type targetP)
{
	if (LCLP < targetP)
	{
		// LCL level is higher than requested pressure, only dry lift is needed
		return DryLift_<Type>(P, T, targetP);
	}

	// Wanted height is above LCL
	if (P < LCLP)
	{
		// Current level is above LCL, only moist lift is required
		return MoistLiftA_<Type>(P, T, targetP);
	}

	// First lift dry adiabatically to LCL height
	const Type LCLT = DryLift_<Type>(P, T, LCLP);

	// Lift from LCL to wanted pressure
	return MoistLiftA_<Type>(LCLP, LCLT, targetP);
}

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

template <typename Type>
CUDA_DEVICE double Lift_(Type P, Type T, Type TD, Type targetP)
{
	lcl_t<Type> LCL = metutil::LCLA_<Type>(P, T, TD);

	if (LCL.P < targetP)
	{
		// LCL level is higher than requested pressure, only dry lift is needed
		return DryLift_<Type>(P, T, targetP);
	}

	return MoistLift_<Type>(P, T, targetP);
}

namespace smarttool
{
/**
 * @brief Calculate equivalent potential temperature.
 *
 * Original:
 * double NFmiSoundingFunctions::CalcThetaE(double T, double Td, double P)
 *
 * Function has been modified so that it takes humidity as an argument;
 * the original function took dewpoint and calculated humidity from that.
 */

template <typename Type>
CUDA_DEVICE Type ThetaE_(Type T, Type RH, Type P)
{
	ASSERT(RH >= 0);
	ASSERT(RH < 102);
	ASSERT(T > 150 || IsMissing(T));
	ASSERT(T < 350 || IsMissing(T));
	ASSERT(P > 1500);

	const Type tpot = himan::metutil::Theta_<Type>(T, P);
	const Type w = himan::metutil::smarttool::MixingRatio_<Type>(T, RH, P);
	return tpot + 3 * w;
}

}  // namespace smarttool
}  // namespace metutil
}  // namespace himan
