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

namespace himan
{

namespace plugin
{

namespace dewpoint_cuda
{

__global__ void UnpackAndCalculate(const unsigned char* dTPacked, 
									const unsigned char* dRHPacked,
									double* dT,
									double* dRH,
									double* dTOut,
									dewpoint_cuda_options opts, int* dMissingValuesCount);

__global__ void Calculate(const double* dT, const double* dRH, double* dTD, dewpoint_cuda_options, int* dMissingValueCount);

__device__ void _Calculate(const double* __restrict__ dT, const double* __restrict__ dRH, double* __restrict__ dTD, dewpoint_cuda_options opts, int* dMissingValueCount, int idx);

} // namespace dewpoint
} // namespace plugin
} // namespace himan

__global__ void himan::plugin::dewpoint_cuda::UnpackAndCalculate(const unsigned char* dTPacked,
									const unsigned char* dRHPacked,
									double* dT,
									double* dRH,
									double* dTDOut,
									dewpoint_cuda_options opts, int* dMissingValuesCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		SimpleUnpack(dTPacked, dT, opts.N, opts.simplePackedT.bitsPerValue, opts.simplePackedT.binaryScaleFactor, opts.simplePackedT.decimalScaleFactor, opts.simplePackedT.referenceValue, idx);
		SimpleUnpack(dRHPacked, dRH, opts.N, opts.simplePackedRH.bitsPerValue, opts.simplePackedRH.binaryScaleFactor, opts.simplePackedRH.decimalScaleFactor, opts.simplePackedRH.referenceValue, idx);

		_Calculate(dT, dRH, dTDOut, opts, dMissingValuesCount, idx);
	}
}

__global__ void himan::plugin::dewpoint_cuda::Calculate(const double* dT, const double* dRH, double* dTDOut, dewpoint_cuda_options opts, int* dMissingValuesCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		_Calculate(dT, dRH, dTDOut, opts, dMissingValuesCount, idx);
	}
}

__device__ void himan::plugin::dewpoint_cuda::_Calculate(const double* __restrict__ dT,
																const double* __restrict__ dRH,
																double* __restrict__ dTD, dewpoint_cuda_options opts,
																int* dMissingValuesCount, int idx)
{

	if (dT[idx] == kFloatMissing || dRH[idx] == kFloatMissing)
	{
		atomicAdd(dMissingValuesCount, 1);
		dTD[idx] = kFloatMissing;
	}
	else
	{
		dTD[idx] = ((dT[idx]+opts.TBase) / (1 - ((dT[idx]+opts.TBase) * log(dRH[idx]) * (RW_div_L)))) - 273.15 + opts.TBase;
	}
}

void himan::plugin::dewpoint_cuda::DoCuda(dewpoint_cuda_options& opts)
{

	CUDA_CHECK(cudaSetDevice(opts.cudaDeviceIndex));

	size_t memSize = opts.N * sizeof(double);

	// Allocate device arrays

	double* dT;
	double* dRH;
	double* dTD;

	unsigned char* dTPacked;
	unsigned char* dRHPacked;
	
	int* dMissingValuesCount;

	CUDA_CHECK(cudaMalloc((void **) &dMissingValuesCount, sizeof(int)));

	CUDA_CHECK(cudaMalloc((void **) &dT, memSize));
	CUDA_CHECK(cudaMalloc((void **) &dRH, memSize));
	CUDA_CHECK(cudaMalloc((void **) &dTD, memSize));

	if (opts.simplePackedT.HasData())
	{
		CUDA_CHECK(cudaMalloc((void **) &dTPacked, opts.simplePackedT.dataLength * sizeof(unsigned char)));
		CUDA_CHECK(cudaMemcpy(dTPacked, opts.simplePackedT.data, opts.simplePackedT.dataLength * sizeof(unsigned char), cudaMemcpyHostToDevice));
	}
	else
	{
		CUDA_CHECK(cudaMemcpy(dT, opts.TIn, memSize, cudaMemcpyHostToDevice));		
	}

	if (opts.simplePackedRH.HasData())
	{
		CUDA_CHECK(cudaMalloc((void **) &dRHPacked, opts.simplePackedRH.dataLength * sizeof(unsigned char)));
		CUDA_CHECK(cudaMemcpy(dRHPacked, opts.simplePackedRH.data, opts.simplePackedRH.dataLength * sizeof(unsigned char), cudaMemcpyHostToDevice));
	}
	else
	{
		CUDA_CHECK(cudaMemcpy(dRH, opts.RHIn, memSize, cudaMemcpyHostToDevice));
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
		UnpackAndCalculate <<< gridDim, blockDim >>> (dTPacked, dRHPacked, dT, dRH, dTD, opts, dMissingValuesCount);
	}
	else
	{
		Calculate <<< gridDim, blockDim >>> (dT, dRH, dTD, opts, dMissingValuesCount);
	}
	

	// block until the device has completed
	CUDA_CHECK(cudaDeviceSynchronize());

	// check if kernel execution generated an error

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	// Retrieve result from device
	CUDA_CHECK(cudaMemcpy(opts.TDOut, dTD, memSize, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(&opts.missingValuesCount, dMissingValuesCount, sizeof(int), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(dT));
	CUDA_CHECK(cudaFree(dRH));
	CUDA_CHECK(cudaFree(dTD));
	CUDA_CHECK(cudaFree(dMissingValuesCount));

	if (opts.simplePackedT.HasData())
	{
		CUDA_CHECK(cudaFree(dTPacked));
	}

	if (opts.simplePackedRH.HasData())
	{
		CUDA_CHECK(cudaFree(dRHPacked));
	}

}
