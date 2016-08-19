/**
 * @file   stability_cuda.h
 * @author partio
 *
 * @date March 14, 2013, 2:17 PM
 */

#ifndef STABILITY_CUDA_H
#define STABILITY_CUDA_H

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
	info_simple* bs01;
	info_simple* bs06;

	double* t500m;
	double* td500m;
	double* p500m;

	double* u01;
	double* v01;
	double* u06;
	double* v06;

	size_t missing;
	size_t N;

	options()
	    : t500(0),
	      t700(0),
	      t850(0),
	      td500(0),
	      td700(0),
	      td850(0),
	      ki(0),
	      si(0),
	      li(0),
	      vti(0),
	      cti(0),
	      tti(0),
	      bs01(0),
	      bs06(0),
	      t500m(0),
	      td500m(0),
	      p500m(0),
	      u01(0),
	      v01(0),
	      u06(0),
	      v06(0),
	      missing(0),
	      N(0)
	{
	}
};

void Process(options& opts);

#ifdef __CUDACC__
void Prepare(const double* source, double** devptr, size_t memsize, cudaStream_t& stream);

__global__ void Calculate(cdarr_t d_t850, cdarr_t d_t700, cdarr_t d_t500, cdarr_t d_td850, cdarr_t d_td700,
                          cdarr_t d_t500m, cdarr_t d_td500m, cdarr_t d_p500m, cdarr_t d_u01, cdarr_t d_v01,
                          cdarr_t d_u06, cdarr_t d_v06, darr_t d_ki, darr_t d_vti, darr_t d_cti, darr_t d_tti,
                          darr_t d_si, darr_t d_li, darr_t d_bs01, darr_t d_bs06, options opts);
#endif

}  // namespace stability_cuda
}  // namespace plugin
}  // namespace himan

#endif /* HAVE_CUDA */

#endif /* STABILITY_CUDA_H */
