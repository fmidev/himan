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

__global__ void Calculate(const double* __restrict__ dT,
							const double* __restrict__ dP,
							double* __restrict__ dTp,
							tpot_cuda_options opts,
							int* dMissingValuesCount);

} // namespace tpot
} // namespace plugin
} // namespace himan

__global__ void himan::plugin::tpot_cuda::Calculate(const double* __restrict__ dT,
													const double* __restrict__ dP,
													double* __restrict__ dTp,
													tpot_cuda_options opts,
													int* dMissingValuesCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		double P = (opts.isConstantPressure) ? opts.PConst : dP[idx];

		if (dT[idx] == kFloatMissing || P == kFloatMissing)
		{
			atomicAdd(dMissingValuesCount, 1);
			dTp[idx] = kFloatMissing;
		}
		else
		{
			dTp[idx] = (opts.TBase + dT[idx]) * powf((1000 / (opts.PScale * P)), 0.28586);
		}
	}
}

void himan::plugin::tpot_cuda::DoCuda(tpot_cuda_options& opts, tpot_cuda_data& datas)
{

	CUDA_CHECK(cudaSetDevice(opts.cudaDeviceIndex));

	size_t memsize = opts.N * sizeof(double);

	// Allocate device arrays

	unsigned char* dpT;
	unsigned char* dpP;
	int* dbmT;
	int* dbmP;

	double* dT;
	double* dP;
	double* dTp;

	int *dMissingValuesCount;

	CUDA_CHECK(cudaMalloc((void **) &dMissingValuesCount, sizeof(int)));

	CUDA_CHECK(cudaHostGetDevicePointer(&dTp, datas.Tp, 0));

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

	if (!opts.isConstantPressure)
	{
		if (opts.pP)
		{
			CUDA_CHECK(cudaHostGetDevicePointer(&dP, datas.P, 0));
			CUDA_CHECK(cudaHostGetDevicePointer(&dpP, datas.pP.data, 0));

			if (datas.pP.HasBitmap())
			{
				CUDA_CHECK(cudaHostGetDevicePointer(&dbmP, datas.pP.bitmap, 0));
			}
		}
		else
		{
			CUDA_CHECK(cudaMalloc((void **) &dP, memsize));
			CUDA_CHECK(cudaMemcpy(dP, datas.P, memsize, cudaMemcpyHostToDevice));
		}
	}

	int src=0;

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
	if (opts.pP)
	{
		SimpleUnpack <<< gridDim, blockDim >>> (dpP, dP, dbmP, datas.pP.coefficients, opts.N, datas.pP.HasBitmap());
	}

	Calculate <<< gridDim, blockDim >>> (dT, dP, dTp, opts, dMissingValuesCount);

	// block until the device has completed
	CUDA_CHECK(cudaDeviceSynchronize());

	// check if kernel execution generated an error

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	CUDA_CHECK(cudaMemcpy(&opts.missingValuesCount, dMissingValuesCount, sizeof(int), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(dMissingValuesCount));

	if (!opts.pT)
	{
		CUDA_CHECK(cudaFree(dT));
	}
	
	if (!opts.isConstantPressure && !opts.pP)
	{
		CUDA_CHECK(cudaFree(dP));
	}
}
