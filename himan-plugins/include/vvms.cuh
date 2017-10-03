/**
 * File:   vvms_cuda.h
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
	double vv_ms_scale;

	options()
	    : t(0),
	      p(0),
	      vv(0),
	      vv_ms(0),
	      N(0),
	      p_const(0),
	      t_base(0),
	      p_scale(1),
	      is_constant_pressure(false),
	      vv_ms_scale(1)
	{
	}
};

void Process(options& opts);

#ifdef __CUDACC__
__global__ void Calculate(cdarr_t d_t, cdarr_t d_vv, cdarr_t d_p, darr_t d_vv_ms, options opts);
#endif

}  // namespace vvms_cuda
}  // namespace plugin
}  // namespace himan

#endif /* HAVE_CUDA */
#endif /* VVMS_CUDA_H */
