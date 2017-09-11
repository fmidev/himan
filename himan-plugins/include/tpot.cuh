/**
 * File:   tpot_cuda.h
 *
 * List of extern functions compiled by nvcc for plugins (compiled by gcc)
 */

#ifndef TPOT_CUDA_H
#define TPOT_CUDA_H

#ifdef HAVE_CUDA

#include "info_simple.h"

namespace himan
{
namespace plugin
{
namespace tpot_cuda
{
struct options
{
	info_simple* t;
	info_simple* td;
	info_simple* p;
	info_simple* tp;
	info_simple* tpw;
	info_simple* tpe;

	size_t N;
	size_t missing;
	double t_base;
	double td_base;
	double p_scale;
	double p_const;
	bool is_constant_pressure;
	bool theta;
	bool thetaw;
	bool thetae;

	options()
	    : t(0),
	      td(0),
	      p(0),
	      tp(0),
	      tpw(0),
	      tpe(0),
	      N(0),
	      missing(0),
	      t_base(0),
	      td_base(0),
	      p_scale(1),
	      p_const(0),
	      is_constant_pressure(false),
	      theta(false),
	      thetaw(false),
	      thetae(false)
	{
	}
};

void Process(options& options);

#ifdef __CUDACC__
__global__ void Calculate(const double* __restrict__ d_t, const double* __restrict__ d_p,
                          const double* __restrict__ d_td, double* __restrict__ d_tp, double* __restrict__ d_tpw,
                          double* __restrict__ d_tpe, options opts);
__device__ double Theta(double T, double P, options opts);
__device__ double ThetaW(double T, double P, double TD, options opts);
__device__ double ThetaE(double T, double P, double TD, options opts);
#endif

}  // namespace tpot_cuda
}  // namespace plugin
}  // namespace himan

#endif /* HAVE_CUDA */
#endif /* TPOT_CUDA_H */
