#include "cuda_plugin_helper.h"

#include "metutil.h"
#include "tpot.cuh"

__global__ void himan::plugin::tpot_cuda::Calculate(const double* __restrict__ d_t, const double* __restrict__ d_p,
                                                    const double* __restrict__ d_td, double* __restrict__ d_tp,
                                                    double* __restrict__ d_tpw, double* __restrict__ d_tpe,
                                                    options opts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		double P = (opts.is_constant_pressure) ? opts.p_const : d_p[idx];

		if (opts.theta)
		{
			d_tp[idx] = Theta(opts.t_base + d_t[idx], P * opts.p_scale, opts);
		}
		if (opts.thetaw)
		{
			d_tpw[idx] = ThetaW(opts.t_base + d_t[idx], opts.p_scale * P, opts.td_base + d_td[idx], opts);
		}
		if (opts.thetae)
		{
			d_tpe[idx] = ThetaE(opts.t_base + d_t[idx], opts.p_scale * P, opts.td_base + d_td[idx], opts);
		}
	}
}
__device__ double himan::plugin::tpot_cuda::Theta(double T, double P, options opts) { return metutil::Theta_(T, P); }
__device__ double himan::plugin::tpot_cuda::ThetaW(double T, double P, double TD, options opts)
{
	double thetaE = ThetaE(T, P, TD, opts);

	return metutil::ThetaW_(thetaE);
}

__device__ double himan::plugin::tpot_cuda::ThetaE(double T, double P, double TD, options opts)
{
	return metutil::ThetaE_(T, TD, P);
}

void himan::plugin::tpot_cuda::Process(options& opts)
{
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	double* d_t = 0;
	double* d_p = 0;
	double* d_td = 0;
	double* d_tp = 0;
	double* d_tpw = 0;
	double* d_tpe = 0;

	size_t memsize = opts.N * sizeof(double);

	// dims

	const int blockSize = 512;
	const int gridSize = opts.N / blockSize + (opts.N % blockSize == 0 ? 0 : 1);

	// Allocate memory on device

	if (opts.theta)
	{
		CUDA_CHECK(cudaMalloc((void**)&d_tp, memsize));
		PrepareInfo(opts.tp);
	}

	if (opts.thetaw)
	{
		CUDA_CHECK(cudaMalloc((void**)&d_tpw, memsize));
		PrepareInfo(opts.tpw);
	}

	if (opts.thetae)
	{
		CUDA_CHECK(cudaMalloc((void**)&d_tpe, memsize));
		PrepareInfo(opts.tpe);
	}

	CUDA_CHECK(cudaMalloc((void**)&d_t, memsize));

	PrepareInfo(opts.t, d_t, stream);

	if (!opts.is_constant_pressure)
	{
		CUDA_CHECK(cudaMalloc((void**)&d_p, memsize));

		PrepareInfo(opts.p, d_p, stream);
	}

	// td

	if (opts.thetaw || opts.thetae)
	{
		CUDA_CHECK(cudaMalloc((void**)&d_td, memsize));

		PrepareInfo(opts.td, d_td, stream);
	}

	CUDA_CHECK(cudaStreamSynchronize(stream));

	Calculate<<<gridSize, blockSize, 0, stream>>>(d_t, d_p, d_td, d_tp, d_tpw, d_tpe, opts);

	// block until the device has completed
	CUDA_CHECK(cudaStreamSynchronize(stream));

	// check if kernel execution generated an error

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	// Retrieve result from device

	if (opts.theta)
	{
		ReleaseInfo(opts.tp, d_tp, stream);
		CUDA_CHECK(cudaFree(d_tp));
	}

	if (opts.thetaw)
	{
		ReleaseInfo(opts.tpw, d_tpw, stream);
		CUDA_CHECK(cudaFree(d_tpw));
	}

	if (opts.thetae)
	{
		ReleaseInfo(opts.tpe, d_tpe, stream);
		CUDA_CHECK(cudaFree(d_tpe));
	}

	CUDA_CHECK(cudaFree(d_t));
	ReleaseInfo(opts.t);

	if (d_p)
	{
		CUDA_CHECK(cudaFree(d_p));
		ReleaseInfo(opts.p);
	}

	if (d_td)
	{
		CUDA_CHECK(cudaFree(d_td));
		ReleaseInfo(opts.td);
	}

	CUDA_CHECK(cudaStreamDestroy(stream));
}
