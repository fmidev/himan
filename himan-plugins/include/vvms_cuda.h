/**
 * File:   vvms_cuda.h
 * Author: partio
 *
 * Created on February 17, 2013, 3:32 PMi
 * 
 * List of extern functions compiled by nvcc for plugins (compiled by gcc)
 */

#ifndef VVMS_CUDA_H
#define VVMS_CUDA_H
#ifdef HAVE_CUDA

#include "info_simple.h"

namespace himan
{
namespace plugin
{
namespace vvms_cuda
{

struct options
{
	info_simple* t;
	info_simple* p;
	info_simple* vv;
	info_simple* vv_ms;

	size_t N;
	double p_const;
	double t_base;
	double p_scale;
	bool is_constant_pressure;
	size_t missing;
	double vv_ms_scale;
	
	options() : N(0), p_const(0), t_base(0), p_scale(1), is_constant_pressure(false), missing(0), vv_ms_scale(1) {}
};

void Process(options& opts);

#ifdef __CUDACC__
__global__ void Calculate(const double* __restrict__ d_t, const double* __restrict__ d_vv, const double* __restrict__ d_p, double* __restrict__ d_vv_ms,
							options opts, int* d_missing);
#endif

} // namespace vvms_cuda
} // namespace plugin
} // namespace himan

#endif	/* HAVE_CUDA */
#endif	/* VVMS_CUDA_H */

