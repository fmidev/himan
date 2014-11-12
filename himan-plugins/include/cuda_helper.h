#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <iostream>
#include <cassert>

#ifdef HAVE_CUDA

#include "info_simple.h"

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

typedef const double* __restrict__ cdarr_t;
typedef double* __restrict__ darr_t;

void CheckCudaError(cudaError_t errarg, const char* file, const int line);
void CheckCudaErrorString(const char* errstr, const char* file,	const int line);

#define CUDA_CHECK(errarg)	 himan::CheckCudaError(errarg, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR_MSG(errstr) himan::CheckCudaErrorString(errstr, __FILE__, __LINE__)

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

#ifdef __NVCC__

inline
void PrepareInfo(info_simple* source)
{
	size_t memsize = source->size_x * source->size_y * sizeof(double);

	assert(source->values);

	CUDA_CHECK(cudaHostRegister(reinterpret_cast <void*> (source->values), memsize, 0));

}

inline
void PrepareInfo(info_simple* source, double* devptr, cudaStream_t& stream)
{
	PrepareInfo(source);

	assert(devptr);
	
	size_t memsize = source->size_x * source->size_y * sizeof(double);
	
	if (source->packed_values)
	{
		// Unpack data and copy it back to host, we need it because its put back to cache
		source->packed_values->Unpack(devptr, source->size_x * source->size_y, &stream);
		CUDA_CHECK(cudaMemcpyAsync(source->values, devptr, memsize, cudaMemcpyDeviceToHost, stream));
	}
	else
	{
		CUDA_CHECK(cudaMemcpyAsync(devptr, source->values, memsize, cudaMemcpyHostToDevice, stream));
	}
}

inline
void ReleaseInfo(info_simple* source)
{
	CUDA_CHECK(cudaHostUnregister(source->values));
}

inline
void ReleaseInfo(info_simple* source, double *devptr, cudaStream_t& stream)
{
	CUDA_CHECK(cudaMemcpyAsync(source->values, devptr, source->size_x * source->size_y * sizeof(double), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));
	ReleaseInfo(source);
}


template<typename T>
__global__
void Fill(T* devptr, size_t N, T fillValue)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		devptr[idx] = fillValue;
	}
}

#endif

} // namespace himan
#if 0
#include "packed_data.h"

namespace himan
{

inline __host__ void UnpackBitmap(const unsigned char* __restrict__ bitmap, int* unpacked, size_t len)
{
	size_t i, v = 1, idx = 0;
	short j = 0;

	for (i = 0; i < len; i++)
	{
		for (j = 7; j >= 0; j--)
		{
			if (BitTest(bitmap[i], j))
			{
				unpacked[idx] = v++;
			}
      idx++;
    }
  }
}

} // namespace himan
#endif
#endif // HAVE_CUDA
#endif // CUDA_HELPER_H
