/**
 * @file metutil.cpp
 *
 * @brief Different utility functions and classes in a namespace
 *
 * @date Apr 29, 2014
 * @author partio
 */

#include "metutil.h"

using namespace himan;
using namespace std;

void metutil::MixingRatio(cdarr_t T, cdarr_t P, darr_t result, size_t N)
{
	for (size_t i = 0; i < N; i++)
	{
		result[i] = MixingRatio_(T[i], P[i]);
	}
}

void metutil::DryLift(cdarr_t P, cdarr_t T, darr_t result, double targetP, size_t N)
{
	for (size_t i = 0; i < N; i++)
	{
		result[i] = DryLift_(P[i], T[i], targetP);
	}
}

void metutil::MoistLift(cdarr_t P, cdarr_t T, cdarr_t TD, darr_t result, double targetP, size_t N)
{
	for (size_t i = 0; i < N; i++)
	{
		result[i] = MoistLift_(P[i], T[i], TD[i], targetP);
	}
}

