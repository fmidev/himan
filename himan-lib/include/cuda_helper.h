#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cassert>
#include <iostream>

#ifdef __CUDACC__
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#define CUDA_KERNEL __global__
#define CUDA_INLINE __forceinline__
#else
#define CUDA_HOST
#define CUDA_DEVICE
#define CUDA_KERNEL
#define CUDA_INLINE
#endif

typedef const double* __restrict__ cdarr_t;
typedef double* __restrict__ darr_t;

#ifdef HAVE_CUDA

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <cuda_runtime.h>
#pragma GCC diagnostic pop

namespace himan
{
/*
 * Two very commonly used data types with cuda calculations.
 * By defining double pointers __restrict__ we guarantee to the compiler
 * that the pointers do not overlap in memory and enable the compiler to
 * generate more efficient code. By adding the const specifier we can
 * benefit from cuda read-only memory which is supposed to be (according
 * to nvidia) faster than regular global memory.
 */

void CheckCudaError(cudaError_t errarg, const char* file, const int line);
void CheckCudaErrorString(const char* errstr, const char* file, const int line);

#define CUDA_CHECK(errarg) himan::CheckCudaError(errarg, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR_MSG(errstr) himan::CheckCudaErrorString(errstr, __FILE__, __LINE__)

inline void CheckCudaError(cudaError_t errarg, const char* file, const int line)
{
	if (errarg)
	{
		std::cerr << "Error at " << file << "(" << line << "): " << cudaGetErrorString(errarg) << std::endl;
		abort();
	}
}

inline void CheckCudaErrorString(const char* errstr, const char* file, const int line)
{
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		std::cerr << "Error: " << errstr << " " << file << " at (" << line << "): " << cudaGetErrorString(err)
		          << std::endl;

		abort();
	}
}

}  // namespace himan

#endif  // HAVE_CUDA
#endif  // CUDA_HELPER_H
