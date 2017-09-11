// System includes
#include <iostream>
#include <string>

#include "cuda_plugin_helper.h"
#include "metutil.h"
#include "relative_humidity.cuh"

// CUDA-kernel that computes RH from T and TD
__global__ void himan::plugin::relative_humidity_cuda::CalculateTTD(cdarr_t d_T, cdarr_t d_TD, darr_t d_RH,
                                                                    options opts)
{
	const double b = 17.27;
	const double c = 237.3;
	const double d = 1.8;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		d_RH[idx] = MissingDouble();

		if (!IsMissingDouble(d_T[idx]) && !IsMissingDouble(d_TD[idx]))
		{
			double td = d_TD[idx] + opts.TDBase - constants::kKelvin;
			double t = d_T[idx] + opts.TBase - constants::kKelvin;

			d_RH[idx] = exp(d + b * (td / (td + c))) / exp(d + b * (t / (t + c)));
			d_RH[idx] = fmax(fmin(1.0, d_RH[idx]), 0.0) * 100.0;
		}
	}
}

// CUDA-kernel that computes RH from T, Q and P
__global__ void himan::plugin::relative_humidity_cuda::CalculateTQP(cdarr_t d_T, cdarr_t d_Q, cdarr_t d_P, darr_t d_RH,
                                                                    options opts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		d_RH[idx] = MissingDouble();

		if (!IsMissingDouble(d_T[idx]) && !IsMissingDouble(d_Q[idx]) && !IsMissingDouble(d_P[idx]))
		{
			double p = d_P[idx] * opts.PScale;
			double ES = himan::metutil::Es_(d_T[idx]) * 0.01;

			d_RH[idx] = (p * d_Q[idx] / constants::kEp / ES) * (p - ES) / (p - d_Q[idx] * p / constants::kEp);
			d_RH[idx] = fmax(fmin(1.0, d_RH[idx]), 0.0) * 100.0;
		}
	}
}

// CUDA-kernel that computes RH on pressure-level from T and Q
__global__ void himan::plugin::relative_humidity_cuda::CalculateTQ(cdarr_t d_T, cdarr_t d_Q, darr_t d_RH, options opts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		d_RH[idx] = MissingDouble();

		if (!IsMissingDouble(d_T[idx]) && !IsMissingDouble(d_Q[idx]))
		{
			double ES = himan::metutil::Es_(d_T[idx]) * 0.01;

			d_RH[idx] = (opts.P_level * d_Q[idx] / constants::kEp / ES) * (opts.P_level - ES) /
			            (opts.P_level - d_Q[idx] * opts.P_level / constants::kEp);
			d_RH[idx] = fmax(fmin(1.0, d_RH[idx]), 0.0) * 100.0;
		}
	}
}

void himan::plugin::relative_humidity_cuda::Process(options& opts)
{
	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	size_t memsize = opts.N * sizeof(double);

	// Define device arrays

	double* d_RH = 0;
	double* d_T = 0;

	// Allocate memory on device

	CUDA_CHECK(cudaMalloc((void**)&d_RH, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_T, memsize));

	// Copy data to device

	PrepareInfo(opts.T, d_T, stream);
	PrepareInfo(opts.RH);

	// dims

	const int blockSize = 512;
	const int gridSize = opts.N / blockSize + (opts.N % blockSize == 0 ? 0 : 1);

	// Select mode in which RH is calculated (with T-TD, T-Q-P or T-Q)
	switch (opts.select_case)
	{
		// Case where RH is calculated from T and TD
		case 0:
		{
			// Define device arrays
			double* d_TD = 0;

			// Allocate memory on device

			CUDA_CHECK(cudaMalloc((void**)&d_TD, memsize));

			// Copy data to device

			PrepareInfo(opts.TD, d_TD, stream);

			CalculateTTD<<<gridSize, blockSize, 0, stream>>>(d_T, d_TD, d_RH, opts);

			// block until the stream has completed
			CUDA_CHECK(cudaStreamSynchronize(stream));

			// check if kernel execution generated an error
			CUDA_CHECK_ERROR_MSG("Kernel invocation");

			ReleaseInfo(opts.TD);

			// Free device memory

			CUDA_CHECK(cudaFree(d_TD));

			break;
		}
		// Case where RH is calculated from T, Q and P
		case 1:
		{
			// Define device arrays

			double* d_Q = 0;
			double* d_P = 0;

			// Allocate memory on device
			CUDA_CHECK(cudaMalloc((void**)&d_Q, memsize));
			CUDA_CHECK(cudaMalloc((void**)&d_P, memsize));

			// Copy data to device
			PrepareInfo(opts.Q, d_Q, stream);
			PrepareInfo(opts.P, d_P, stream);

			CalculateTQP<<<gridSize, blockSize, 0, stream>>>(d_T, d_Q, d_P, d_RH, opts);

			// block until the stream has completed
			CUDA_CHECK(cudaStreamSynchronize(stream));

			// check if kernel execution generated an error
			CUDA_CHECK_ERROR_MSG("Kernel invocation");

			// Retrieve result from device
			ReleaseInfo(opts.Q);
			ReleaseInfo(opts.P);

			// Free device memory

			CUDA_CHECK(cudaFree(d_Q));
			CUDA_CHECK(cudaFree(d_P));

			break;
		}
		// Case where RH is calculated for pressure levels from T and Q
		case 2:
		{
			// Define device arrays
			double* d_Q = 0;

			// Allocate memory on device
			CUDA_CHECK(cudaMalloc((void**)&d_Q, memsize));

			// Copy data to device
			PrepareInfo(opts.Q, d_Q, stream);

			CalculateTQ<<<gridSize, blockSize, 0, stream>>>(d_T, d_Q, d_RH, opts);

			// block until the stream has completed
			CUDA_CHECK(cudaStreamSynchronize(stream));

			// check if kernel execution generated an error
			CUDA_CHECK_ERROR_MSG("Kernel invocation");

			// Retrieve result from device
			ReleaseInfo(opts.Q);

			// Free device memory

			CUDA_CHECK(cudaFree(d_Q));

			break;
		}
	}

	ReleaseInfo(opts.T);
	ReleaseInfo(opts.RH, d_RH, stream);

	CUDA_CHECK(cudaFree(d_T));
	CUDA_CHECK(cudaFree(d_RH));

	cudaStreamDestroy(stream);
}
