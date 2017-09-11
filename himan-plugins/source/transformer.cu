#include "cuda_plugin_helper.h"
#include "transformer.cuh"

__global__ void himan::plugin::transformer_cuda::Calculate(cdarr_t d_source, darr_t d_dest, options opts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		d_dest[idx] = __fma_rn(d_source[idx], opts.scale, opts.base);
	}
}

void himan::plugin::transformer_cuda::Process(options& opts)
{
	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	size_t memsize = opts.N * sizeof(double);

	// Allocate device arrays

	double *d_source = 0, *d_dest = 0;

	// Allocate memory on device

	CUDA_CHECK(cudaMalloc((void**)&d_source, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_dest, memsize));

	// Copy data to device

	PrepareInfo(opts.source, d_source, stream);
	PrepareInfo(opts.dest, d_dest, stream);

	// dims

	const int blockSize = 512;
	const int gridSize = opts.N / blockSize + (opts.N % blockSize == 0 ? 0 : 1);

	CUDA_CHECK(cudaStreamSynchronize(stream));

	Calculate<<<gridSize, blockSize, 0, stream>>>(d_source, d_dest, opts);

	// block until the stream has completed
	CUDA_CHECK(cudaStreamSynchronize(stream));

	// check if kernel execution generated an error
	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	// Retrieve result from device

	// Free device memory

	himan::ReleaseInfo(opts.source);
	himan::ReleaseInfo(opts.dest, d_dest, stream);

	CUDA_CHECK(cudaFree(d_source));
	CUDA_CHECK(cudaFree(d_dest));

	cudaStreamDestroy(stream);
}
