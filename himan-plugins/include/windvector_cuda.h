/**
 * @file   windvector_cuda.h
 * @author partio
 *
 * @date March 14, 2013, 2:17 PM
 */

#ifndef WINDVECTOR_CUDA_H
#define	WINDVECTOR_CUDA_H

#ifdef HAVE_CUDA

#include "info_simple.h"

namespace himan
{
namespace plugin
{

enum HPWindVectorTargetType
{
	kUnknownElement = 0,
	kWind,
	kGust,
	kSea,
	kIce
};

namespace windvector_cuda
{

struct options
{
	info_simple* u;
	info_simple* v;
	info_simple* speed;
	info_simple* dir;
	info_simple* vector;

	HPWindVectorTargetType target_type;
	bool vector_calculation;
	bool need_grid_rotation;
	size_t missing;
	size_t N;

	options() 
		: target_type(kUnknownElement)
		, vector_calculation(false)
		, need_grid_rotation(false)
		, missing(0)
		, N(0)
	{}

	
};


void Process(options& opts);


#ifdef __CUDACC__
__global__ void Calculate(const double* __restrict__ d_u, const double* __restrict__ d_v, double* __restrict__ d_speed, double* __restrict__ d_dir, double* __restrict__ d_vector,
							options opts, int* d_missing);
__global__ void Rotate(double* __restrict__ dU, double* __restrict__ dV, info_simple opts);

#endif

} // namespace windvector_cuda
} // namespace plugin
} // namespace himan

#endif  /* HAVE_CUDA */
#endif	/* WINDVECTOR_CUDA_H */
