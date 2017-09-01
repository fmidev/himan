#include "cuda_plugin_helper.h"
#include "dewpoint.cuh"
#include "metutil.h"

__global__ void himan::plugin::dewpoint_cuda::Calculate(cdarr_t d_t, cdarr_t d_rh, darr_t d_td, options opts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		double RH = d_rh[idx] * opts.rh_scale;
		d_td[idx] = metutil::DewPointFromRH_(d_t[idx] + opts.t_base, RH);
	}
}

void himan::plugin::dewpoint_cuda::Process(options& opts)
{
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	size_t memsize = opts.N * sizeof(double);

	// Allocate device arrays

	double* d_t = 0;
	double* d_rh = 0;
	double* d_td = 0;

	// Allocate memory on device

	CUDA_CHECK(cudaMalloc((void**)&d_t, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_rh, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_td, memsize));

	himan::PrepareInfo(opts.t, d_t, stream);
	himan::PrepareInfo(opts.rh, d_rh, stream);
	himan::PrepareInfo(opts.td);

	// dims

	const int blockSize = 512;
	const int gridSize = opts.N / blockSize + (opts.N % blockSize == 0 ? 0 : 1);

	CUDA_CHECK(cudaStreamSynchronize(stream));

	Calculate<<<gridSize, blockSize, 0, stream>>>(d_t, d_rh, d_td, opts);

	CUDA_CHECK(cudaStreamSynchronize(stream));
	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	himan::ReleaseInfo(opts.t);
	himan::ReleaseInfo(opts.rh);
	himan::ReleaseInfo(opts.td, d_td, stream);

	CUDA_CHECK(cudaStreamSynchronize(stream));

	// Free device memory

	CUDA_CHECK(cudaFree(d_t));
	CUDA_CHECK(cudaFree(d_td));
	CUDA_CHECK(cudaFree(d_rh));

	CUDA_CHECK(cudaStreamDestroy(stream));
}
