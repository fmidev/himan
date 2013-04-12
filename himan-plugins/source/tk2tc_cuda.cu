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

__global__ void UnpackAndCalculate(const unsigned char* dTPacked, double* dT, double* dTOut, tk2tc_cuda_options opts, int* dMissingValuesCount);
__global__ void Calculate(const double* dT, double* dTOut, tk2tc_cuda_options opts, int* dMissingValuesCount);

__device__ void _Calculate(const double* __restrict__ dT, double* __restrict__ dTOut, tk2tc_cuda_options opts, int* dMissingValuesCount);

} // namespace tk2tc_cuda
} // namespace plugin
} // namespace himan

__global__ void himan::plugin::tk2tc_cuda::UnpackAndCalculate(const unsigned char* dTPacked, double* dT, double* dTOut, tk2tc_cuda_options opts, int* dMissingValuesCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		if (opts.simplePackedT.bitsPerValue%8)
		{
			SimpleUnpackUnevenBytes(dTPacked, dT, opts.N, opts.simplePackedT.bitsPerValue, opts.simplePackedT.binaryScaleFactor, opts.simplePackedT.decimalScaleFactor, opts.simplePackedT.referenceValue);
		}
		else
		{
			SimpleUnpackFullBytes(dTPacked, dT, opts.N, opts.simplePackedT.bitsPerValue, opts.simplePackedT.binaryScaleFactor, opts.simplePackedT.decimalScaleFactor, opts.simplePackedT.referenceValue);
		}
		
		_Calculate(dT, dTOut, opts, dMissingValuesCount);
	}
}

__global__ void himan::plugin::tk2tc_cuda::Calculate(const double* dT, double* dTOut, tk2tc_cuda_options opts, int* dMissingValuesCount)
{
	_Calculate(dT, dTOut, opts, dMissingValuesCount);
}

__device__ void himan::plugin::tk2tc_cuda::_Calculate(const double* __restrict__ dT, double* __restrict__ dTOut, tk2tc_cuda_options opts, int* dMissingValuesCount)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		if (dT[idx] == kFloatMissing)
		{
			atomicAdd(dMissingValuesCount, 1);
			dTOut[idx] = kFloatMissing;
		}
		else
		{
			dTOut[idx] = dT[idx] - 273.15;
		}
	}
}

void himan::plugin::tk2tc_cuda::DoCuda(tk2tc_cuda_options& opts)
{

	CUDA_CHECK(cudaSetDevice(opts.cudaDeviceIndex));

	size_t memSize = opts.N * sizeof(double);

	// Allocate device arrays

	double* dT;
	unsigned char* dTPacked;
	double *dTOut;

	int *dMissingValuesCount;
	
	CUDA_CHECK(cudaMalloc((void **) &dT, memSize));
	CUDA_CHECK(cudaMalloc((void **) &dTOut, memSize));
	CUDA_CHECK(cudaMalloc((void **) &dMissingValuesCount, sizeof(int)));

	if (opts.isPackedData)
	{
		CUDA_CHECK(cudaMalloc((void **) &dTPacked, opts.simplePackedT.dataLength * sizeof(unsigned char)));
		CUDA_CHECK(cudaMemcpy(dTPacked, opts.simplePackedT.data, opts.simplePackedT.dataLength * sizeof(unsigned char), cudaMemcpyHostToDevice));

		//opts.simplePackedT.Clear();
	}
	else
	{
		CUDA_CHECK(cudaMemcpy(dT, opts.TIn, memSize, cudaMemcpyHostToDevice));
	}

	int src = 0;
	
	CUDA_CHECK(cudaMemcpy(dMissingValuesCount, &src, sizeof(int), cudaMemcpyHostToDevice));

	// dims

	const int blockSize = 512;
	const int gridSize = opts.N/blockSize + (opts.N%blockSize == 0?0:1);

	dim3 gridDim(gridSize);
	dim3 blockDim(blockSize);

	if (opts.isPackedData)
	{
		UnpackAndCalculate <<< gridDim, blockDim >>> (dTPacked, dT, dTOut, opts, dMissingValuesCount);
	}
	else
	{
		Calculate <<< gridDim, blockDim >>> (dT, dTOut, opts, dMissingValuesCount);
	}

	// block until the device has completed
	CUDA_CHECK(cudaDeviceSynchronize());

	// check if kernel execution generated an error

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	// Retrieve result from device
	CUDA_CHECK(cudaMemcpy(opts.TOut, dTOut, memSize, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(&opts.missingValuesCount, dMissingValuesCount, sizeof(int), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(dT));
	CUDA_CHECK(cudaFree(dTOut));
	CUDA_CHECK(cudaFree(dMissingValuesCount));

	if (opts.isPackedData)
	{
		CUDA_CHECK(cudaFree(dTPacked));
	}


}
