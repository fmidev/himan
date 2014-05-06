// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "cuda_helper.h"
#include "relative_humidity_cuda.h"

// CUDA-kernel that computes RH from T and TD
__global__ void himan::plugin::relative_humidity_cuda::CalculateTTD(double* __restrict__ d_T,
														double* __restrict__ d_TD,
														double* __restrict__ d_RH,
														options opts,
														int* d_missing)
{
	const double b = 17.27;
	const double c = 237.3;
	const double d = 1.8;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		if (d_T[idx] == kFloatMissing || d_TD[idx] == kFloatMissing)
		{
			atomicAdd(d_missing, 1);
			d_RH[idx] = kFloatMissing;
		}
		else
		{
			d_TD[idx] += opts.TDBase;
			d_T[idx] += opts.TBase;
			d_RH[idx] = exp(d + b * (d_TD[idx] / (d_TD[idx] + c))) / exp(d + b * (d_T[idx] / (d_T[idx] + c)));
			d_RH[idx] = fmax(fmin(1.0,d_RH[idx]),0.0)*100.0;
		}
	}
}

// CUDA-kernel that computes RH from T, Q and P
__global__ void himan::plugin::relative_humidity_cuda::CalculateTQP(double* __restrict__ d_T,
														const double* __restrict__ d_Q,
														double* __restrict__ d_P,
														double* __restrict__ d_ES,
														double* __restrict__ d_RH,
														options opts,
														int* d_missing)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		if (d_T[idx] == kFloatMissing || d_Q[idx] == kFloatMissing || d_P[idx] == kFloatMissing)
		{
			atomicAdd(d_missing, 1);
			d_RH[idx] = kFloatMissing;
		}
		else
		{
			d_P[idx] *= opts.PScale;
			d_T[idx] += opts.TBase;

			if (d_T[idx] > -5.0)
			{
				d_ES[idx] = 6.107 * exp10(7.5 * d_T[idx] / (237.0 + d_T[idx]));
			}
			else
			{
				d_ES[idx] = 6.107 * exp10(9.5 * d_T[idx] / (265.5 + d_T[idx]));
			}

			d_RH[idx] = (d_P[idx] * d_Q[idx] / opts.kEp / d_ES[idx]) * (d_P[idx] - d_ES[idx]) / (d_P[idx] - d_Q[idx] * d_P[idx] / opts.kEp);
			d_RH[idx] = fmax(fmin(1.0,d_RH[idx]),0.0)*100.0;
		}
	}
}

// CUDA-kernel that computes RH on pressure-level from T and Q
__global__ void himan::plugin::relative_humidity_cuda::CalculateTQ(double* __restrict__ d_T,
														const double* __restrict__ d_Q,
														double* __restrict__ d_ES,
														double* __restrict__ d_RH,
														options opts,
														int* d_missing)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		if (d_T[idx] == kFloatMissing || d_Q[idx] == kFloatMissing)
		{
			atomicAdd(d_missing, 1);
			d_RH[idx] = kFloatMissing;
		}
		else
		{
			d_T[idx] += opts.TBase;

			if (d_T[idx] > -5.0)
			{
				d_ES[idx] = 6.107 * exp10(7.5 * d_T[idx] / (237.0 + d_T[idx]));
			}
			else
			{
				d_ES[idx] = 6.107 * exp10(9.5 * d_T[idx] / (265.5 + d_T[idx]));
			}

			d_RH[idx] = (opts.P_level * d_Q[idx] / opts.kEp / d_ES[idx]) * (opts.P_level - d_ES[idx]) / (opts.P_level - d_Q[idx] * opts.P_level / opts.kEp);
			d_RH[idx] = fmax(fmin(1.0,d_RH[idx]),0.0)*100.0;
		}
	}
}

void himan::plugin::relative_humidity_cuda::Process(options& opts)
{

	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	size_t memsize = opts.N * sizeof(double);

	// Select mode in which RH is calculated (with T-TD, T-Q-P or T-Q)
	switch (opts.select_case)
	{
	// Case where RH is calculated from T and TD
	case 0:	
	{
		// Allocate device arrays

		double* d_T = 0;
		double* d_TD = 0;
		double* d_RH = 0;
		int* d_missing = 0;

		// Allocate memory on device
		CUDA_CHECK(cudaMalloc((void **) &d_T, memsize));
		CUDA_CHECK(cudaMalloc((void **) &d_TD, memsize));
		CUDA_CHECK(cudaMalloc((void **) &d_RH, memsize));
		CUDA_CHECK(cudaMalloc((void **) &d_missing, sizeof(int)));

		// Copy data to device

		if (opts.T->packed_values)
		{
			// Unpack data and copy it back to host, we need it because its put back to cache
			d_T = opts.T->packed_values->Unpack(&stream);
			CUDA_CHECK(cudaMemcpyAsync(opts.T->values, d_T, memsize, cudaMemcpyDeviceToHost, stream));
		}
		else
		{
			CUDA_CHECK(cudaMemcpyAsync(d_T, opts.T->values, memsize, cudaMemcpyHostToDevice, stream));
		}
	
		if (opts.TD->packed_values)
		{
			// Unpack data and copy it back to host, we need it because its put back to cache
			d_TD = opts.TD->packed_values->Unpack(&stream);
			CUDA_CHECK(cudaMemcpyAsync(opts.TD->values, d_TD, memsize, cudaMemcpyDeviceToHost, stream));
		}
		else
		{
			CUDA_CHECK(cudaMemcpyAsync(d_TD, opts.TD->values, memsize, cudaMemcpyHostToDevice, stream));
		}
		int src = 0;
	
		CUDA_CHECK(cudaMemcpyAsync(d_missing, &src, sizeof(int), cudaMemcpyHostToDevice, stream));

		// dims

		const int blockSize = 512;
		const int gridSize = opts.N/blockSize + (opts.N%blockSize == 0?0:1);
	
		CUDA_CHECK(cudaStreamSynchronize(stream));
	
		CalculateTTD <<< gridSize, blockSize, 0, stream >>> (d_T, d_TD, d_RH, opts, d_missing);
	
		// block until the stream has completed
		CUDA_CHECK(cudaStreamSynchronize(stream));

		// check if kernel execution generated an error
		CUDA_CHECK_ERROR_MSG("Kernel invocation");

		// Retrieve result from device
		CUDA_CHECK(cudaMemcpyAsync(&opts.missing, d_missing, sizeof(int), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(opts.RH->values, d_RH, memsize, cudaMemcpyDeviceToHost, stream));

		CUDA_CHECK(cudaStreamSynchronize(stream));

		// Free device memory

		CUDA_CHECK(cudaFree(d_T));
		CUDA_CHECK(cudaFree(d_TD));
		CUDA_CHECK(cudaFree(d_RH));
		CUDA_CHECK(cudaFree(d_missing));

		break;
	}
	// Case where RH is calculated from T, Q and P
	case 1:
	{
		// Allocate device arrays

		double* d_T = 0;
		double* d_Q = 0;
		double* d_P = 0;
		double* d_ES = 0;
		double* d_RH = 0;
		int* d_missing = 0;

		// Allocate memory on device
		CUDA_CHECK(cudaMalloc((void **) &d_T, memsize));
		CUDA_CHECK(cudaMalloc((void **) &d_Q, memsize));
		CUDA_CHECK(cudaMalloc((void **) &d_P, memsize));
		CUDA_CHECK(cudaMalloc((void **) &d_ES, memsize));
		CUDA_CHECK(cudaMalloc((void **) &d_RH, memsize));
		CUDA_CHECK(cudaMalloc((void **) &d_missing, sizeof(int)));

		// Copy data to device

		if (opts.T->packed_values)
		{
			// Unpack data and copy it back to host, we need it because its put back to cache
			d_T = opts.T->packed_values->Unpack(&stream);
			CUDA_CHECK(cudaMemcpyAsync(opts.T->values, d_T, memsize, cudaMemcpyDeviceToHost, stream));
		}
		else
		{
			CUDA_CHECK(cudaMemcpyAsync(d_T, opts.T->values, memsize, cudaMemcpyHostToDevice, stream));
		}

		if (opts.Q->packed_values)
		{
			// Unpack data and copy it back to host, we need it because its put back to cache
			d_Q = opts.Q->packed_values->Unpack(&stream);
			CUDA_CHECK(cudaMemcpyAsync(opts.Q->values, d_Q, memsize, cudaMemcpyDeviceToHost, stream));
		}
		else
		{
			CUDA_CHECK(cudaMemcpyAsync(d_Q, opts.Q->values, memsize, cudaMemcpyHostToDevice, stream));
		}

		if (opts.P->packed_values)
		{
			// Unpack data and copy it back to host, we need it because its put back to cache
			d_P = opts.P->packed_values->Unpack(&stream);
			CUDA_CHECK(cudaMemcpyAsync(opts.P->values, d_P, memsize, cudaMemcpyDeviceToHost, stream));
		}
		else
		{
			CUDA_CHECK(cudaMemcpyAsync(d_P, opts.P->values, memsize, cudaMemcpyHostToDevice, stream));
		}

		int src = 0;
	
		CUDA_CHECK(cudaMemcpyAsync(d_missing, &src, sizeof(int), cudaMemcpyHostToDevice, stream));

		// dims

		const int blockSize = 512;
		const int gridSize = opts.N/blockSize + (opts.N%blockSize == 0?0:1);
	
		CUDA_CHECK(cudaStreamSynchronize(stream));

		CalculateTQP <<< gridSize, blockSize, 0, stream >>> (d_T, d_Q, d_P, d_ES, d_RH, opts, d_missing);

		// block until the stream has completed
		CUDA_CHECK(cudaStreamSynchronize(stream));

		// check if kernel execution generated an error
		CUDA_CHECK_ERROR_MSG("Kernel invocation");

		// Retrieve result from device
		CUDA_CHECK(cudaMemcpyAsync(&opts.missing, d_missing, sizeof(int), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(opts.RH->values, d_RH, memsize, cudaMemcpyDeviceToHost, stream));

		CUDA_CHECK(cudaStreamSynchronize(stream));

		// Free device memory

		CUDA_CHECK(cudaFree(d_T));
		CUDA_CHECK(cudaFree(d_Q));
		CUDA_CHECK(cudaFree(d_P));
		CUDA_CHECK(cudaFree(d_ES));
		CUDA_CHECK(cudaFree(d_RH));
		CUDA_CHECK(cudaFree(d_missing));

		break;
	}
	// Case where RH is calculated for pressure levels from T and Q
	case 2:
	{
		// Allocate device arrays

		double* d_T = 0;
		double* d_Q = 0;
		double* d_RH = 0;
		double* d_ES = 0;	
		int* d_missing = 0;

		// Allocate memory on device
		CUDA_CHECK(cudaMalloc((void **) &d_T, memsize));
		CUDA_CHECK(cudaMalloc((void **) &d_Q, memsize));
		CUDA_CHECK(cudaMalloc((void **) &d_ES, memsize));
		CUDA_CHECK(cudaMalloc((void **) &d_RH, memsize));
		CUDA_CHECK(cudaMalloc((void **) &d_missing, sizeof(int)));

		// Copy data to device

		if (opts.T->packed_values)
		{
			// Unpack data and copy it back to host, we need it because its put back to cache
			d_T = opts.T->packed_values->Unpack(&stream);
			CUDA_CHECK(cudaMemcpyAsync(opts.T->values, d_T, memsize, cudaMemcpyDeviceToHost, stream));
		}
		else
		{
			CUDA_CHECK(cudaMemcpyAsync(d_T, opts.T->values, memsize, cudaMemcpyHostToDevice, stream));
		}

		if (opts.Q->packed_values)
		{
			// Unpack data and copy it back to host, we need it because its put back to cache
			d_Q = opts.Q->packed_values->Unpack(&stream);
			CUDA_CHECK(cudaMemcpyAsync(opts.Q->values, d_Q, memsize, cudaMemcpyDeviceToHost, stream));
		}
		else
		{
			CUDA_CHECK(cudaMemcpyAsync(d_Q, opts.Q->values, memsize, cudaMemcpyHostToDevice, stream));
		}
		int src = 0;
	
		CUDA_CHECK(cudaMemcpyAsync(d_missing, &src, sizeof(int), cudaMemcpyHostToDevice, stream));

		// dims

		const int blockSize = 512;
		const int gridSize = opts.N/blockSize + (opts.N%blockSize == 0?0:1);

		CUDA_CHECK(cudaStreamSynchronize(stream));

		CalculateTQ <<< gridSize, blockSize, 0, stream >>> (d_T, d_Q, d_ES, d_RH, opts, d_missing);

		// block until the stream has completed
		CUDA_CHECK(cudaStreamSynchronize(stream));

		// check if kernel execution generated an error
		CUDA_CHECK_ERROR_MSG("Kernel invocation");

		// Retrieve result from device
		CUDA_CHECK(cudaMemcpyAsync(&opts.missing, d_missing, sizeof(int), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(opts.RH->values, d_RH, memsize, cudaMemcpyDeviceToHost, stream));

		CUDA_CHECK(cudaStreamSynchronize(stream));

		// Free device memory
	
		CUDA_CHECK(cudaFree(d_T));
		CUDA_CHECK(cudaFree(d_Q));
		CUDA_CHECK(cudaFree(d_ES));
		CUDA_CHECK(cudaFree(d_RH));
		CUDA_CHECK(cudaFree(d_missing));

		break;
		}
	}
    cudaStreamDestroy(stream);

}
