/**
 * @file   relative_humidity_cuda.h
 * @author Tack
 *
 * @date April 22, 2014
 */

#ifndef RELATIVE_HUMIDITY_CUDA_H
#define RELATIVE_HUMIDITY_CUDA_H

#ifdef HAVE_CUDA

#include "info_simple.h"

namespace himan
{
namespace plugin
{
namespace relative_humidity_cuda
{

struct options
{
	info_simple* T;
	info_simple* TD;
	info_simple* Q;
	info_simple* P;
	info_simple* RH;

	int select_case;
	size_t N;
	size_t missing;
	double kEp;
	double TDBase;
	double TBase;
	double PScale;
	double P_level;


	options() : select_case(1), N(0), missing(0), kEp(0), TDBase(0), TBase(0), PScale(1), P_level(0) {}
};

void Process(options& opts);

#ifdef __CUDACC__
__global__ void CalculateTTD(double* __restrict__ d_T, double* __restrict__ d_TD, double* __restrict__ d_RH, options opts, int* d_missing);
__global__ void CalculateTQP(double* __restrict__ d_T, const double* __restrict__ d_Q, double* __restrict__ d_P, double* __restrict__ d_ES, double* __restrict__ d_RH, options opts, int* d_missing);
__global__ void CalculateTQ(double* __restrict__ d_T, const double* __restrict__ d_Q, double* __restrict__ d_ES, double* __restrict__ d_RH, options opts, int* d_missing);

#endif

} // namespace relative_humidity_cuda
} // namespace plugin
} // namespace himan

#endif  /* HAVE_CUDA */
#endif	/* RELATIVE_HUMIDITY_CUDA_H */

