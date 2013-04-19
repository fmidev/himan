// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "vvms_cuda.h"
#include "cuda_helper.h"

namespace himan
{

namespace plugin
{

namespace vvms_cuda
{

__global__ void UnpackAndCalculate(const unsigned char* dTPacked,
									const unsigned char* dVVPacked,
									const unsigned char* dPPacked,
									double*  dT,
									double* dVV,
									double* dP,
									double* dVVOut,
									vvms_cuda_options opts,
									int* dMissingValuesCount);

__global__ void Calculate(const double* dT,
							const double* dVV,
							const double* dP,
							double* dVVOut,
							vvms_cuda_options opts,
							int* dMissingValuesCount);

__device__ void _Calculate(const double* __restrict__ dT,
							const double* __restrict__ dVV,
							const double* __restrict__ dP,
							double* __restrict__ dVVOut,
							vvms_cuda_options opts,
							int* dMissingValuesCount,
							int idx);

} // namespace vvms_cuda
} // namespace plugin
} // namespace himan

__global__ void himan::plugin::vvms_cuda::UnpackAndCalculate(const unsigned char* dTPacked,
									const unsigned char* dVVPacked,
									const unsigned char* dPPacked,
									double* dT,
									double* dVV,
									double* dP,
									double* dVVOut,
									vvms_cuda_options opts,
									int* dMissingValuesCount)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < opts.N)
	{
		if (opts.simplePackedT.HasData())
		{
			SimpleUnpack(dTPacked, dT, opts.N, opts.simplePackedT.bitsPerValue, opts.simplePackedT.binaryScaleFactor, opts.simplePackedT.decimalScaleFactor, opts.simplePackedT.referenceValue, idx);
		}

		if (opts.simplePackedVV.HasData())
		{
			SimpleUnpack(dVVPacked, dVV, opts.N, opts.simplePackedVV.bitsPerValue, opts.simplePackedVV.binaryScaleFactor, opts.simplePackedVV.decimalScaleFactor, opts.simplePackedVV.referenceValue, idx);
		}

		if (!opts.isConstantPressure && opts.simplePackedP.HasData())
		{
			SimpleUnpack(dPPacked, dP, opts.N, opts.simplePackedP.bitsPerValue, opts.simplePackedP.binaryScaleFactor, opts.simplePackedP.decimalScaleFactor, opts.simplePackedP.referenceValue, idx);
		}

		_Calculate(dT, dVV, dP, dVVOut, opts, dMissingValuesCount, idx);
	}
}

__device__ void himan::plugin::vvms_cuda::_Calculate(const double* __restrict__ dT,
														const double* __restrict__ dVV,
														const double* __restrict__ dP,
														double* __restrict__ VVOut,
														vvms_cuda_options opts, int* dMissingValuesCount, int idx)
{

	double P = (opts.isConstantPressure) ? opts.PConst : dP[idx];

	if (dT[idx] == kFloatMissing || dVV[idx] == kFloatMissing || P == kFloatMissing)
	{
		atomicAdd(dMissingValuesCount, 1);
		VVOut[idx] = kFloatMissing;
	}
	else
	{
		VVOut[idx] = 287 * -dVV[idx] * (opts.TBase + dT[idx]) / (9.81 * P);
	}
}

__global__ void himan::plugin::vvms_cuda::Calculate(const double* dT, 
														const double* dVV,
														const double* dP,
														double* dVVOut,
														vvms_cuda_options opts, int* dMissingValuesCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		_Calculate(dT, dVV, dP, dVVOut, opts, dMissingValuesCount, idx);
	}
	
}

void himan::plugin::vvms_cuda::DoCuda(vvms_cuda_options& opts)
{

	CUDA_CHECK(cudaSetDevice(opts.cudaDeviceIndex));

	size_t memSize = opts.N * sizeof(double);

	// Allocate device arrays

	double* dT = NULL;
	double* dP = NULL;
	double* dVV = NULL;
	double* dVVOut = NULL;

	unsigned char* dTPacked = NULL;
	unsigned char* dPPacked = NULL;
	unsigned char* dVVPacked = NULL;

	int *dMissingValuesCount = NULL;

	CUDA_CHECK(cudaMalloc((void **) &dMissingValuesCount, sizeof(int)));
	CUDA_CHECK(cudaMalloc((void **) &dT, memSize));
	CUDA_CHECK(cudaMalloc((void **) &dVV, memSize));
	CUDA_CHECK(cudaMalloc((void **) &dVVOut, memSize));

	if (opts.simplePackedT.HasData())
	{
		CUDA_CHECK(cudaMalloc((void **) &dTPacked, opts.simplePackedT.dataLength * sizeof(unsigned char)));
		CUDA_CHECK(cudaMemcpy(dTPacked, opts.simplePackedT.data, opts.simplePackedT.dataLength * sizeof(unsigned char), cudaMemcpyHostToDevice));
	}
	else
	{
		CUDA_CHECK(cudaMemcpy(dT, opts.TIn, memSize, cudaMemcpyHostToDevice));
	}

	if (opts.simplePackedVV.HasData())
	{
		CUDA_CHECK(cudaMalloc((void **) &dVVPacked, opts.simplePackedVV.dataLength * sizeof(unsigned char)));
		CUDA_CHECK(cudaMemcpy(dVVPacked, opts.simplePackedVV.data, opts.simplePackedVV.dataLength * sizeof(unsigned char), cudaMemcpyHostToDevice));
	}
	else
	{
		CUDA_CHECK(cudaMemcpy(dVV, opts.VVIn, memSize, cudaMemcpyHostToDevice));
	}

	if (!opts.isConstantPressure)
	{
		CUDA_CHECK(cudaMalloc((void **) &dP, memSize));

		if (opts.simplePackedP.HasData())
		{
			CUDA_CHECK(cudaMalloc((void **) &dPPacked, opts.simplePackedP.dataLength * sizeof(unsigned char)));
			CUDA_CHECK(cudaMemcpy(dPPacked, opts.simplePackedP.data, opts.simplePackedP.dataLength * sizeof(unsigned char), cudaMemcpyHostToDevice));
		}
		else
		{
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
		UnpackAndCalculate <<< gridDim, blockDim >>> (dTPacked, dVVPacked, dPPacked, dT, dVV, dP, dVVOut, opts, dMissingValuesCount);
	}
	else
	{
		Calculate <<< gridDim, blockDim >>> (dT, dVV, dP, dVVOut, opts, dMissingValuesCount);
	}

	// block until the device has completed
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	// Retrieve result from device
	CUDA_CHECK(cudaMemcpy(opts.VVOut, dVVOut, memSize, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(&opts.missingValuesCount, dMissingValuesCount, sizeof(int), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(dT));
	CUDA_CHECK(cudaFree(dVV));
	CUDA_CHECK(cudaFree(dVVOut));

	CUDA_CHECK(cudaFree(dMissingValuesCount));

	if (opts.simplePackedT.HasData())
	{
		CUDA_CHECK(cudaFree(dTPacked));
	}

	if (opts.simplePackedVV.HasData())
	{
		CUDA_CHECK(cudaFree(dVVPacked));
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
