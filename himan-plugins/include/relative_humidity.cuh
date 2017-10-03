/**
 * @file   relative_humidity_cuda.h
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
	double TDBase;
	double TBase;
	double PScale;
	double P_level;

	options()
	    : T(0), TD(0), Q(0), P(0), RH(0), select_case(1), N(0), missing(0), TDBase(0), TBase(0), PScale(1), P_level(0)
	{
	}
};

void Process(options& opts);

#ifdef __CUDACC__
__global__ void CalculateTTD(cdarr_t d_T, cdarr_t d_TD, darr_t d_RH, options opts);
__global__ void CalculateTQP(cdarr_t d_T, cdarr_t d_Q, cdarr_t d_P, darr_t d_RH, options opts);
__global__ void CalculateTQ(cdarr_t d_T, cdarr_t d_Q, darr_t d_RH, options opts);

#endif

}  // namespace relative_humidity_cuda

}  // namespace plugin
}  // namespace himan

#endif /* HAVE_CUDA */
#endif /* RELATIVE_HUMIDITY_CUDA_H */
