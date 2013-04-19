// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "cuda_helper.h"
#include "tpot_cuda.h"

namespace himan
{

namespace plugin
{

namespace tpot_cuda
{

__global__ void UnpackAndCalculate(const unsigned char* dTPacked,
									const unsigned char* dPPacked,
									double* dT,
									double* dP,
									double* dTP,
									tpot_cuda_options opts, int* dMissingValuesCount);

__global__ void Calculate(double* dT,
							double* dP,
							double* dTP,
							tpot_cuda_options opts, int* dMissingValuesCount);

__device__ void _Calculate(const double* __restrict__ dT,
							const double* __restrict__ dP,
							double* __restrict__ dTP,
							tpot_cuda_options opts, 
							int* dMissingValuesCount,
							int idx);

} // namespace tpot
} // namespace plugin
} // namespace himan

__global__ void himan::plugin::tpot_cuda::UnpackAndCalculate(const unsigned char* dTPacked,
									const unsigned char* dPPacked,
									double* dT,
									double* dP,
									double* dTP,
									tpot_cuda_options opts, int* dMissingValuesCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		if (opts.simplePackedT.HasData())
		{
			SimpleUnpack(dTPacked, dT, opts.N, opts.simplePackedT.bitsPerValue, opts.simplePackedT.binaryScaleFactor, opts.simplePackedT.decimalScaleFactor, opts.simplePackedT.referenceValue, idx);
		}

		if (!opts.isConstantPressure && opts.simplePackedP.HasData())
		{
			SimpleUnpack(dPPacked, dP, opts.N, opts.simplePackedP.bitsPerValue, opts.simplePackedP.binaryScaleFactor, opts.simplePackedP.decimalScaleFactor, opts.simplePackedP.referenceValue, idx);
		}

		_Calculate(dT, dP, dTP, opts, dMissingValuesCount, idx);
	}
}

__global__ void himan::plugin::tpot_cuda::Calculate(double* dT,	double* dP, double* dTP,
							tpot_cuda_options opts, int* dMissingValuesCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		_Calculate(dT, dP, dTP, opts, dMissingValuesCount, idx);
	}
}

__device__ void himan::plugin::tpot_cuda::_Calculate(const double* __restrict__ dT,
														const double* __restrict__ dP,
														double* __restrict__ dTP,
														tpot_cuda_options opts,
														int* dMissingValuesCount, int idx)
{
	double P = (opts.isConstantPressure) ? opts.PConst : dP[idx];

	if (dT[idx] == kFloatMissing || P == kFloatMissing)
	{
		atomicAdd(dMissingValuesCount, 1);
		dTP[idx] = kFloatMissing;
	}
	else
	{
		dTP[idx] = (opts.TBase + dT[idx]) * powf((1000 / (opts.PScale * P)), 0.286);
	}
}

void himan::plugin::tpot_cuda::DoCuda(tpot_cuda_options& opts)
{

	CUDA_CHECK(cudaSetDevice(opts.cudaDeviceIndex));

	size_t memSize = opts.N * sizeof(double);

	// Allocate device arrays

	unsigned char* dTPacked;
	unsigned char* dPPacked;

	double* dT;
	double* dP;
	double* dTP;

	int *dMissingValuesCount;

	CUDA_CHECK(cudaMalloc((void **) &dMissingValuesCount, sizeof(int)));

	CUDA_CHECK(cudaMalloc((void **) &dT, memSize));
	CUDA_CHECK(cudaMalloc((void **) &dTP, memSize));

	if (opts.simplePackedT.HasData())
	{
		CUDA_CHECK(cudaMalloc((void **) &dTPacked, opts.N * sizeof(unsigned char)));
		CUDA_CHECK(cudaMemcpy(dTPacked, opts.simplePackedT.data, opts.N * sizeof(unsigned char), cudaMemcpyHostToDevice));
	}
	else
	{
		CUDA_CHECK(cudaMemcpy(dT, opts.TIn, memSize, cudaMemcpyHostToDevice));
	}

	if (!opts.isConstantPressure)
	{
		if (opts.simplePackedT.HasData())
		{
			CUDA_CHECK(cudaMalloc((void **) &dPPacked, opts.N * sizeof(unsigned char)));
			CUDA_CHECK(cudaMemcpy(dPPacked, opts.simplePackedP.data, opts.N * sizeof(unsigned char), cudaMemcpyHostToDevice));
		}
		else
		{
			CUDA_CHECK(cudaMalloc((void **) &dP, memSize));
			CUDA_CHECK(cudaMemcpy(dP, opts.PIn, memSize, cudaMemcpyHostToDevice));
		}
	}

	int src=0;

	CUDA_CHECK(cudaMemcpy(dMissingValuesCount, &src, sizeof(int), cudaMemcpyHostToDevice));

	// dims

	const int blockSize = 512;
	const int gridSize = opts.N/blockSize + (opts.N%blockSize == 0?0:1);

	dim3 gridDim(gridSize);
	dim3 blockDim(blockSize);

	if (opts.isPackedData)
	{
		UnpackAndCalculate <<< gridDim, blockDim >>> (dTPacked, dPPacked, dT, dP, dTP, opts, dMissingValuesCount);
	}
	else
	{
		Calculate <<< gridDim, blockDim >>> (dT, dP, dTP, opts, dMissingValuesCount);
	}

	// block until the device has completed
	CUDA_CHECK(cudaDeviceSynchronize());

	// check if kernel execution generated an error

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	// Retrieve result from device
	CUDA_CHECK(cudaMemcpy(opts.TpOut, dTP, memSize, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(&opts.missingValuesCount, dMissingValuesCount, sizeof(int), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(dT));
	CUDA_CHECK(cudaFree(dTP));

	CUDA_CHECK(cudaFree(dMissingValuesCount));

	if (opts.simplePackedT.HasData())
	{
		CUDA_CHECK(cudaFree(dTPacked));
	}
	
	if (!opts.isConstantPressure)
	{
		if (opts.simplePackedP.HasData())
		{
			CUDA_CHECK(cudaFree(dPPacked));
		}
		
		CUDA_CHECK(cudaFree(dP));
	}


}
