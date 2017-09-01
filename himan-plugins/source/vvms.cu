#include "cuda_plugin_helper.h"
#include "vvms.cuh"

__global__ void himan::plugin::vvms_cuda::Calculate(cdarr_t d_t, cdarr_t d_vv, cdarr_t d_p, darr_t d_vv_ms,
                                                    options opts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		double P = (opts.is_constant_pressure) ? opts.p_const : d_p[idx];

		d_vv_ms[idx] = opts.vv_ms_scale *
		               (287 * -d_vv[idx] * (opts.t_base + d_t[idx]) / (himan::constants::kG * P * opts.p_scale));
	}
}

void himan::plugin::vvms_cuda::Process(options& opts)
{
	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	size_t memsize = opts.N * sizeof(double);

	// Allocate device arrays

	double* d_t = 0;
	double* d_p = 0;
	double* d_vv = 0;
	double* d_vv_ms = 0;

	CUDA_CHECK(cudaMalloc((void**)&d_vv_ms, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_t, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_vv, memsize));

	PrepareInfo(opts.t, d_t, stream);
	PrepareInfo(opts.vv, d_vv, stream);
	PrepareInfo(opts.vv_ms);

	if (!opts.is_constant_pressure)
	{
		CUDA_CHECK(cudaMalloc((void**)&d_p, memsize));

		PrepareInfo(opts.p, d_p, stream);
	}

	// dims

	const int blockSize = 512;
	const int gridSize = opts.N / blockSize + (opts.N % blockSize == 0 ? 0 : 1);

	CUDA_CHECK(cudaStreamSynchronize(stream));

	Calculate<<<gridSize, blockSize, 0, stream>>>(d_t, d_vv, d_p, d_vv_ms, opts);

	// block until the device has completed
	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	// Retrieve result from device

	himan::ReleaseInfo(opts.vv_ms, d_vv_ms, stream);
	himan::ReleaseInfo(opts.t);
	himan::ReleaseInfo(opts.vv);

	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaFree(d_t));
	CUDA_CHECK(cudaFree(d_vv));
	CUDA_CHECK(cudaFree(d_vv_ms));

	if (d_p)
	{
		himan::ReleaseInfo(opts.p);
		CUDA_CHECK(cudaFree(d_p));
	}

	CUDA_CHECK(cudaStreamDestroy(stream));
}
