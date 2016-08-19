/**
 * @file   dewpoint_cuda.h
 * @author partio
 *
 * @date February 17, 2013, 3:32 PM
 */

#ifndef DEWPOINT_CUDA_H
#define DEWPOINT_CUDA_H

#ifdef HAVE_CUDA

#include "info_simple.h"

namespace himan
{
namespace plugin
{
namespace dewpoint_cuda
{
struct options
{
	info_simple* t;
	info_simple* rh;
	info_simple* td;

	size_t N;
	size_t missing;
	double t_base;
	double rh_scale;

	options() : t(0), rh(0), td(0), N(0), missing(0), t_base(0), rh_scale(1) {}
};

void Process(options& opts);

#ifdef __CUDACC__
__global__ void Calculate(cdarr_t d_t, cdarr_t d_rh, darr_t d_td, options opts);
#endif

}  // namespace dewpoint_cuda
}  // namespace plugin
}  // namespace himan

#endif /* HAVE_CUDA */
#endif /* DEWPOINT_CUDA_H */
