// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "cuda_helper.h"
#include "tk2tc_cuda.h"

namespace himan
{

namespace plugin
{

namespace tk2tc_cuda
{

__global__ void Calculate(const double* __restrict__ dTK, double* __restrict__ dTC, tk2tc_cuda_options opts, int* dMissingValuesCount);

} // namespace tk2tc_cuda
} // namespace plugin
} // namespace himan

__global__ void himan::plugin::tk2tc_cuda::Calculate(const double* __restrict__ dTK,
														double* __restrict__ dTC,
														tk2tc_cuda_options opts,
														int* dMissingValuesCount)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		if (dTK[idx] == kFloatMissing)
		{
			atomicAdd(dMissingValuesCount, 1);
			dTC[idx] = kFloatMissing;
		}
		else
		{
			dTC[idx] = dTK[idx] - 273.15;
		}
	}
}

void himan::plugin::tk2tc_cuda::DoCuda(tk2tc_cuda_options& opts, tk2tc_cuda_data& datas)
{

	CUDA_CHECK(cudaSetDevice(opts.cudaDeviceIndex));
	
	size_t memsize = opts.N * sizeof(double);

	// Allocate device arrays

	double* dTK;
	unsigned char* dpTK;
	int* dbmTK;
	double *dTC;

	int *dMissingValuesCount;
	
	CUDA_CHECK(cudaMalloc((void **) &dMissingValuesCount, sizeof(int)));

	CUDA_CHECK(cudaHostGetDevicePointer(&dTC, datas.TC, 0));

	if (opts.pTK)
	{
		CUDA_CHECK(cudaHostGetDevicePointer(&dTK, datas.TK, 0));
		CUDA_CHECK(cudaHostGetDevicePointer(&dpTK, datas.pTK.data, 0));

		if (datas.pTK.HasBitmap())
		{
			CUDA_CHECK(cudaHostGetDevicePointer(&dbmTK, datas.pTK.bitmap, 0));
		}
	}
	else
	{
		CUDA_CHECK(cudaMalloc((void **) &dTK, memsize));
		CUDA_CHECK(cudaMemcpy(dTK, datas.TK, memsize, cudaMemcpyHostToDevice));
	}

	int src = 0;
	
	CUDA_CHECK(cudaMemcpy(dMissingValuesCount, &src, sizeof(int), cudaMemcpyHostToDevice));

	// dims

	const int blockSize = 512;
	const int gridSize = opts.N/blockSize + (opts.N%blockSize == 0?0:1);

	dim3 gridDim(gridSize);
	dim3 blockDim(blockSize);

	if (opts.pTK)
	{
		SimpleUnpack <<< gridDim, blockDim >>> (dpTK, dTK, dbmTK, datas.pTK.coefficients, opts.N, datas.pTK.HasBitmap());
	}

	Calculate <<< gridDim, blockDim >>> (dTK, dTC, opts, dMissingValuesCount);

	// block until the device has completed
	CUDA_CHECK(cudaDeviceSynchronize());

	// check if kernel execution generated an error

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	// Retrieve result from device
	CUDA_CHECK(cudaMemcpy(&opts.missingValuesCount, dMissingValuesCount, sizeof(int), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(dMissingValuesCount));

	if (!opts.pTK)
	{
		CUDA_CHECK(cudaFree(dTK));
	}

}
