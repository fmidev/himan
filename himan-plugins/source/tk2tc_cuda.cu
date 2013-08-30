// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "cuda_helper.h"
#include "tk2tc_cuda.h"

#include "stdio.h"

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

	int deviceId = (opts.threadIndex <= 32) ? 0 : 1;
	
	CUDA_CHECK(cudaSetDevice(deviceId));

	size_t memsize = opts.N * sizeof(double);

	// Allocate device arrays

	double* dTK = 0, *dTC = 0;

	int *dMissingValuesCount;
	
	CUDA_CHECK(cudaMalloc((void **) &dMissingValuesCount, sizeof(int)));

	CUDA_CHECK(cudaHostGetDevicePointer(&dTC, datas.TC, 0));

	if (opts.pTK)
	{
		CUDA_CHECK(cudaHostGetDevicePointer(&dTK, datas.TK, 0));
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

	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	if (opts.pTK)
	{
		datas.pTK->Unpack(dTK, &stream);
	}

	Calculate <<< gridSize, blockSize, 0, stream >>> (dTK, dTC, opts, dMissingValuesCount);

	// block until the stream has completed
	CUDA_CHECK(cudaStreamSynchronize(stream));

	// check if kernel execution generated an error
	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	// Retrieve missing values from device
	CUDA_CHECK(cudaMemcpy(&opts.missingValuesCount, dMissingValuesCount, sizeof(int), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(dMissingValuesCount));

	if (!opts.pTK)
	{
		CUDA_CHECK(cudaFree(dTK));
	}

    cudaStreamDestroy(stream);

}
