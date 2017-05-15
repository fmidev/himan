/**
 * @file metutil.cpp
 *
 * @brief Different utility functions in a namespace
 *
 */

#include "metutil.h"

using namespace himan;
using namespace std;

void metutil::Gammaw(cdarr_t P, cdarr_t T, darr_t result, size_t N)
{
	for (size_t i = 0; i < N; i++)
	{
		result[i] = Gammaw_(P[i], T[i]);
	}
}

void metutil::MixingRatio(cdarr_t T, cdarr_t P, darr_t result, size_t N)
{
	for (size_t i = 0; i < N; i++)
	{
		result[i] = MixingRatio_(T[i], P[i]);
	}
}

void metutil::DryLift(cdarr_t P, cdarr_t T, cdarr_t targetP, darr_t result, size_t N)
{
	for (size_t i = 0; i < N; i++)
	{
		result[i] = DryLift_(P[i], T[i], targetP[i]);
	}
}

void metutil::MoistLift(cdarr_t P, cdarr_t T, cdarr_t targetP, darr_t result, size_t N)
{
	for (size_t i = 0; i < N; i++)
	{
		result[i] = MoistLift_(P[i], T[i], targetP[i]);
	}
}

void metutil::MoistLiftA(cdarr_t P, cdarr_t T, cdarr_t targetP, darr_t result, size_t N)
{
	for (size_t i = 0; i < N; i++)
	{
		result[i] = MoistLiftA_(P[i], T[i], targetP[i]);
	}
}

void metutil::Lift(cdarr_t P, cdarr_t T, cdarr_t TD, cdarr_t targetP, darr_t result, size_t N)
{
	for (size_t i = 0; i < N; i++)
	{
		result[i] = Lift_(P[i], T[i], TD[i], targetP[i]);
	}
}

void metutil::LiftLCL(cdarr_t P, cdarr_t T, cdarr_t PLCL, cdarr_t targetP, darr_t result, size_t N)
{
	for (size_t i = 0; i < N; i++)
	{
		result[i] = LiftLCL_(P[i], T[i], PLCL[i], targetP[i]);
	}
}

void metutil::Tw(cdarr_t thetaE, cdarr_t P, darr_t result, size_t N)
{
	for (size_t i = 0; i < N; i++)
	{
		result[i] = Tw_(thetaE[i], P[i]);
	}
}

double metutil::WaterProbability_(double T, double RH)
{
	return 1 / (1 + exp(22 - 2.7 * (T - constants::kKelvin) - 0.2 * RH));
}

double metutil::RelativeTopography_(int level1, int level2, double z1, double z2)
{
	int coefficient = 1;
	double topography;
	double height1, height2;

	if (level1 > level2)
	{
		coefficient = -1;
	}

	height1 = z1 * constants::kIg;  // convert to metres z/9.81
	height2 = z2 * constants::kIg;

	topography = coefficient * (height1 - height2);

	return topography;
}

int metutil::LowConvection_(double T0m, double T850)
{
	assert(T0m > 0);
	assert(T850 > 0);

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

double metutil::FlightLevel_(double P)
{
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
		h = (pow(P / PG, -(constants::kRd * gamma) / constants::kG) - 1) * TG / gamma * m_hft;
	}
	// above tropopause
	else
	{
		h = (-log(P / p_tropo) * constants::kRd * T_tropo / constants::kG + h_tropo) * m_hft;
	}

	// round to multiple of 5
	return round(h / 5) * 5;
}
