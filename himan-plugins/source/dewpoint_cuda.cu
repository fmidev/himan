// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "cuda_helper.h"
#include "dewpoint_cuda.h"

#define MAPPED_PINNED

const double RW = 461.5; // Vesihoyryn kaasuvakio (J / K kg)
const double L = 2.5e6; // Veden hoyrystymislampo (J / kg)
const double RW_div_L = RW / L;

namespace himan
{

namespace plugin
{

namespace dewpoint_cuda
{

__global__ void Calculate(const double* __restrict__ dT,
							const double* __restrict__ dRH,
							double* __restrict__ dTD, dewpoint_cuda_options opts, int* dMissingValueCount);

} // namespace dewpoint
} // namespace plugin
} // namespace himan

__global__ void himan::plugin::dewpoint_cuda::Calculate(const double* __restrict__ dT,
															const double* __restrict__ dRH,
															double* __restrict__ dTD, dewpoint_cuda_options opts,
															int* dMissingValuesCount)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
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
}

void himan::plugin::dewpoint_cuda::DoCuda(dewpoint_cuda_options& opts, dewpoint_cuda_data& datas)
{

	CUDA_CHECK(cudaSetDevice(opts.cudaDeviceIndex));

	size_t memsize = opts.N * sizeof(double);

	// Allocate device arrays

	double* dT;
	double* dRH;
	double* dTD;

	unsigned char* dpT;
	unsigned char* dpRH;
	int* dbmT;
	int* dbmRH;
	
	int* dMissingValuesCount;

	CUDA_CHECK(cudaMalloc((void **) &dMissingValuesCount, sizeof(int)));

	CUDA_CHECK(cudaHostGetDevicePointer(&dTD, datas.TD, 0));

	if (opts.pT)
	{
		CUDA_CHECK(cudaHostGetDevicePointer(&dT, datas.T, 0));
		CUDA_CHECK(cudaHostGetDevicePointer(&dpT, datas.pT.data, 0));

		if (datas.pT.HasBitmap())
		{
			CUDA_CHECK(cudaHostGetDevicePointer(&dbmT, datas.pT.bitmap, 0));
		}
	}
	else
	{
		CUDA_CHECK(cudaMalloc((void **) &dT, memsize));
		CUDA_CHECK(cudaMemcpy(dT, datas.T, memsize, cudaMemcpyHostToDevice));
	}

	if (opts.pRH)
	{
		CUDA_CHECK(cudaHostGetDevicePointer(&dRH, datas.RH, 0));
		CUDA_CHECK(cudaHostGetDevicePointer(&dpRH, datas.pRH.data, 0));

		if (datas.pRH.HasBitmap())
		{
			CUDA_CHECK(cudaHostGetDevicePointer(&dbmRH, datas.pRH.bitmap, 0));
		}
	}
	else
	{
		CUDA_CHECK(cudaMalloc((void **) &dRH, memsize));
		CUDA_CHECK(cudaMemcpy(dRH, datas.RH, memsize, cudaMemcpyHostToDevice));
	}

	int src = 0;

	CUDA_CHECK(cudaMemcpy(dMissingValuesCount, &src, sizeof(int), cudaMemcpyHostToDevice));
	
	// dims

	const int blockSize = 512;
	const int gridSize = opts.N/blockSize + (opts.N%blockSize == 0?0:1);

	dim3 gridDim(gridSize);
	dim3 blockDim(blockSize);

	if (opts.pT)
	{
		SimpleUnpack <<< gridDim, blockDim >>> (dpT, dT, dbmT, datas.pT.coefficients, opts.N, datas.pT.HasBitmap());
	}

	if (opts.pRH)
	{
		SimpleUnpack <<< gridDim, blockDim >>> (dpRH, dRH, dbmRH, datas.pRH.coefficients, opts.N, datas.pRH.HasBitmap());
	}

	Calculate <<< gridDim, blockDim >>> (dT, dRH, dTD, opts, dMissingValuesCount);

	// block until the device has completed
	CUDA_CHECK(cudaDeviceSynchronize());

	// check if kernel execution generated an error

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	// Retrieve result from device
	CUDA_CHECK(cudaMemcpy(&opts.missingValuesCount, dMissingValuesCount, sizeof(int), cudaMemcpyDeviceToHost));

	if (!opts.pT)
	{
		CUDA_CHECK(cudaFree(dT));
	}

	if (!opts.pRH)
	{
		CUDA_CHECK(cudaFree(dRH));
	}

	CUDA_CHECK(cudaFree(dMissingValuesCount));

}
