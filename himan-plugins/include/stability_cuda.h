/**
 * @file   stability_cuda.h
 * @author partio
 *
 * @date March 14, 2013, 2:17 PM
 */

#ifndef STABILITY_CUDA_H
#define	STABILITY_CUDA_H

#ifdef HAVE_CUDA
#include "info_simple.h"

namespace himan
{
namespace plugin
{
namespace stability_cuda
{

struct options
{
	info_simple* t500;
	info_simple* t700;
	info_simple* t850;
	info_simple* td500;
	info_simple* td700;
	info_simple* td850;

	info_simple* ki;
	info_simple* si;
	info_simple* li;
	info_simple* vti;
	info_simple* cti;
	info_simple* tti;
	
	size_t missing;
	size_t N;

	options() 
		: t500(0)
		, t700(0)
		, t850(0)
		, td500(0)
		, td700(0)
		, td850(0)
		, ki(0)
		, si(0)
		, li(0)
		, vti(0)
		, cti(0)
		, tti(0)
	{}

};

void Process(options& opts);

#ifdef __CUDACC__
void Prepare(info_simple* source, double* devptr, size_t memsize, cudaStream_t& stream);

__global__ void Calculate(cdarr_t d_t850, cdarr_t d_t700, cdarr_t d_t500, cdarr_t d_td850, cdarr_t d_td700, darr_t d_ki, darr_t d_vti, darr_t d_cti, darr_t d_tti, darr_t d_si, darr_t d_li, options opts, int* d_missing);
__device__ double SI(double T850, double T500, double TD850, int* d_missing);
__device__ double LI(double T500, double T500m, double TD500m, double P500m, int* d_missing);

#endif

} // namespace stability_cuda
} // namespace plugin
} // namespace himan

#endif  /* HAVE_CUDA */

#endif	/* STABILITY_CUDA_H */
