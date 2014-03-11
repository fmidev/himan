// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "cuda_helper.h"
#include "tk2tc_cuda.h"

__global__ void himan::plugin::tk2tc_cuda::Calculate(const double* __restrict__ d_source,
														double* __restrict__ d_dest,
														options opts,
														int* d_missing)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		if (d_source[idx] == kFloatMissing)
		{
			atomicAdd(d_missing, 1);
			d_dest[idx] = kFloatMissing;
		}
		else
		{
			d_dest[idx] = opts.scale * (d_source[idx] + opts.base);
		}
	}
}

void himan::plugin::tk2tc_cuda::Process(options& opts)
{

	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	size_t memsize = opts.N * sizeof(double);

	// Allocate device arrays

	double* d_source = 0, *d_dest = 0;

	int *d_missing;

	// Allocate memory on device

	CUDA_CHECK(cudaMalloc((void **) &d_missing, sizeof(int)));

	CUDA_CHECK(cudaMalloc((void **) &d_source, memsize));
	CUDA_CHECK(cudaMalloc((void **) &d_dest, memsize));

	// Copy data to device

	if (opts.source->packed_values)
	{
		// Unpack data and copy it back to host, we need it because its put back to cache
		d_source = opts.source->packed_values->Unpack(&stream);
		CUDA_CHECK(cudaMemcpyAsync(opts.source->values, d_source, memsize, cudaMemcpyDeviceToHost, stream));
	}
	else
	{
		CUDA_CHECK(cudaMemcpyAsync(d_source, opts.source->values, memsize, cudaMemcpyHostToDevice, stream));
	}

	int src = 0;
	
	CUDA_CHECK(cudaMemcpyAsync(d_missing, &src, sizeof(int), cudaMemcpyHostToDevice, stream));

	// dims

	const int blockSize = 512;
	const int gridSize = opts.N/blockSize + (opts.N%blockSize == 0?0:1);

	CUDA_CHECK(cudaStreamSynchronize(stream));

	Calculate <<< gridSize, blockSize, 0, stream >>> (d_source, d_dest, opts, d_missing);

	// block until the stream has completed
	CUDA_CHECK(cudaStreamSynchronize(stream));

	// check if kernel execution generated an error
	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	// Retrieve result from device
	CUDA_CHECK(cudaMemcpyAsync(&opts.missing, d_missing, sizeof(int), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(opts.dest->values, d_dest, memsize, cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaStreamSynchronize(stream));

	// Free device memory

	CUDA_CHECK(cudaFree(d_source));
	CUDA_CHECK(cudaFree(d_dest));
	CUDA_CHECK(cudaFree(d_missing));

    cudaStreamDestroy(stream);

}
