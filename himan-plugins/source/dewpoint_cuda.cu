// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "cuda_helper.h"
#include "dewpoint_cuda.h"

const double RW = 461.5; // Vesihoyryn kaasuvakio (J / K kg)
const double L = 2.5e6; // Veden hoyrystymislampo (J / kg)
const double RW_div_L = RW / L;
const double K = 273.15;

__global__ void himan::plugin::dewpoint_cuda::Calculate(const double* __restrict__ d_t,
															const double* __restrict__ d_rh,
															double* __restrict__ d_td,
															options opts,
															int* d_missing)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		if (d_t[idx] == kFloatMissing || d_rh[idx] == kFloatMissing)
		{
			atomicAdd(d_missing, 1);
			d_td[idx] = kFloatMissing;
		}
		else
		{
			d_td[idx] = ((d_t[idx]+opts.t_base) / (1 - ((d_t[idx]+opts.t_base) * log(d_rh[idx] * opts.rh_scale) * (RW_div_L)))) - K + opts.t_base;
		}
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
	
	int* d_missing = 0;

	// Allocate memory on device

	CUDA_CHECK(cudaMalloc((void **) &d_missing, sizeof(int)));
	CUDA_CHECK(cudaMalloc((void **) &d_td, memsize));

	if (opts.t->packed_values)
	{
		// Unpack data and copy it back to host, we need it because its put back to cache
		d_t = opts.t->packed_values->Unpack(&stream);
		CUDA_CHECK(cudaMemcpyAsync(opts.t->values, d_t, memsize, cudaMemcpyDeviceToHost, stream));
	}
	else
	{
		CUDA_CHECK(cudaMalloc((void **) &d_t, memsize));
		CUDA_CHECK(cudaMemcpyAsync(d_t, opts.t->values, memsize, cudaMemcpyHostToDevice, stream));
	}

	if (opts.rh->packed_values)
	{
		// Unpack data and copy it back to host, we need it because its put back to cache
		d_rh = opts.rh->packed_values->Unpack(&stream);
		CUDA_CHECK(cudaMemcpyAsync(opts.rh->values, d_rh, memsize, cudaMemcpyDeviceToHost, stream));
	}
	else
	{
		CUDA_CHECK(cudaMalloc((void **) &d_rh, memsize));
		CUDA_CHECK(cudaMemcpyAsync(d_rh, opts.rh->values, memsize, cudaMemcpyHostToDevice, stream));
	}

	int src = 0;

	CUDA_CHECK(cudaMemcpyAsync(d_missing, &src, sizeof(int), cudaMemcpyHostToDevice, stream));
	
	// dims

	const int blockSize = 512;
	const int gridSize = opts.N/blockSize + (opts.N%blockSize == 0?0:1);

	CUDA_CHECK(cudaStreamSynchronize(stream));
	
	Calculate <<< gridSize, blockSize, 0, stream >>> (d_t, d_rh, d_td, opts, d_missing);

	CUDA_CHECK(cudaStreamSynchronize(stream));

	// block until the device has completed

	// check if kernel execution generated an error

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	// Retrieve result from device

	CUDA_CHECK(cudaMemcpyAsync(opts.td->values, d_td, memsize, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(&opts.missing, d_missing, sizeof(int), cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaStreamSynchronize(stream));

	// Free device memory

	CUDA_CHECK(cudaFree(d_t));
	CUDA_CHECK(cudaFree(d_td));
	CUDA_CHECK(cudaFree(d_rh));
	CUDA_CHECK(cudaFree(d_missing));

	CUDA_CHECK(cudaStreamDestroy(stream));
}
