/**
 * File:   tk2tc_cuda.h
 * Author: partio
 *
 * Created on February 17, 2013, 3:32 PM
 * 
 * List of extern functions compiled by nvcc for plugins (compiled by gcc)
 */

#ifndef TK2TC_CUDA_H
#define TK2TC_CUDA_H

#ifdef HAVE_CUDA

#include "simple_packed.h"

namespace himan
{
namespace plugin
{
namespace tk2tc_cuda
{

	void CheckCudaError(cudaError_t errarg, const char* file, const int line);
void CheckCudaErrorString(const char* errstr, const char* file,	const int line);

#define CUDA_CHECK(errarg)	 CheckCudaError(errarg, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR_MSG(errstr) CheckCudaErrorString(errstr, __FILE__, __LINE__)

inline void CheckCudaError(cudaError_t errarg, const char* file, const int line)
{
	if(errarg)
	{
		std::cerr << "Error at " << file << "(" << line << "): " << cudaGetErrorString(errarg) << std::endl;
		exit(1);
	}
}


inline void CheckCudaErrorString(const char* errstr, const char* file,	const int line)
{
	cudaError_t err = cudaGetLastError();

	if(err != cudaSuccess)
	{
		std::cerr	<< "Error: "
					<< errstr
					<< " "
					<< file
					<< " at ("
					<< line
					<< "): "
					<< cudaGetErrorString(err)
					<< std::endl;

		exit(1);
	}
}

struct options
{
	size_t N;
	size_t missing;
	double scale;
	double base;

	double *source;
	double *dest;
	simple_packed* p;

	options() : N(0), missing(0), scale(1), base(0), source(0), dest(0), p(0) {}
};

void Process(options& opts);

__global__ void Calculate(const double* __restrict__ d_source, double* __restrict__ d_dest, options opts, int* d_missing);


} // namespace tk2tc_cuda
} // namespace plugin
} // namespace himan

#endif  /* HAVE_CUDA */
#endif	/* TK2TC_CUDA_H */

