/**
 * @file   cuda_plugin_helper.h
 *
 */

#ifndef CUDA_PLUGIN_HELPER_H
#define CUDA_PLUGIN_HELPER_H

#ifdef HAVE_CUDA

#include "cuda_helper.h"
#include "info_simple.h"
#include "simple_packed.h"

namespace himan
{
#ifdef __NVCC__

inline void PrepareInfo(info_simple* source)
{
	size_t memsize = source->size_x * source->size_y * sizeof(double);

	assert(source->values);

	CUDA_CHECK(cudaHostRegister(reinterpret_cast<void*>(source->values), memsize, 0));
}

inline void PrepareInfo(info_simple* source, double* devptr, cudaStream_t& stream)
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

inline void ReleaseInfo(info_simple* source) { CUDA_CHECK(cudaHostUnregister(source->values)); }
inline void ReleaseInfo(info_simple* source, double* devptr, cudaStream_t& stream)
{
	CUDA_CHECK(cudaMemcpyAsync(source->values, devptr, source->size_x * source->size_y * sizeof(double),
	                           cudaMemcpyDeviceToHost, stream));

	if (0)
	{
		// no bitmap support for now
		bool pack = true;
		for (size_t i = 0; i < source->size_x * source->size_y; i++)
		{
			if (IsMissing(source->values[0]))
			{
				pack = false;
				break;
			}
		}

		if (pack)
		{
			assert(source->packed_values);
			source->packed_values->coefficients.bitsPerValue = 24;

			source->packed_values->Pack(devptr, source->size_x * source->size_y, &stream);
		}
	}

	CUDA_CHECK(cudaStreamSynchronize(stream));
	ReleaseInfo(source);
}

template <typename T>
__global__ void Fill(T* devptr, size_t N, T fillValue)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		devptr[idx] = fillValue;
	}
}

#endif /* __NVCC__ */

}  // namespace himan

#endif /* HAVE_CUDA */

#endif /* CUDA_PLUGIN_HELPER_H */
