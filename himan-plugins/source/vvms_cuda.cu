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

__global__ void Calculate(const double* __restrict__ dT,
							const double* __restrict__ dVV,
							const double* __restrict__ dP,
							double* __restrict__ dVVOut,
							vvms_cuda_options opts,
							int* dMissingValuesCount);

} // namespace vvms_cuda
} // namespace plugin
} // namespace himan


__global__ void himan::plugin::vvms_cuda::Calculate(const double* __restrict__ dT,
														const double* __restrict__ dVV,
														const double* __restrict__ dP,
														double* __restrict__ VVMS,
														vvms_cuda_options opts,
														int* dMissingValuesCount)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		double P = (opts.isConstantPressure) ? opts.PConst : dP[idx];

		if (dT[idx] == kFloatMissing || dVV[idx] == kFloatMissing || P == kFloatMissing)
		{
			atomicAdd(dMissingValuesCount, 1);
			VVMS[idx] = kFloatMissing;
		}
		else
		{
			VVMS[idx] = 287 * -dVV[idx] * (opts.TBase + dT[idx]) / (9.81 * P);
		}
	}
}

void himan::plugin::vvms_cuda::DoCuda(vvms_cuda_options& opts, vvms_cuda_data& datas)
{

	size_t memsize = opts.N * sizeof(double);

	// Allocate device arrays

	double* dT = NULL;
	double* dP = NULL;
	double* dVV = NULL;
	double* dVVMS = NULL;

	int *dMissingValuesCount = NULL;

	CUDA_CHECK(cudaMalloc((void **) &dMissingValuesCount, sizeof(int)));

	CUDA_CHECK(cudaHostGetDevicePointer(&dVVMS, datas.VVMS, 0));

	if (opts.pT)
	{
		CUDA_CHECK(cudaHostGetDevicePointer(&dT, datas.T, 0));

	}
	else
	{
		CUDA_CHECK(cudaMalloc((void **) &dT, memsize));
		CUDA_CHECK(cudaMemcpy(dT, datas.T, memsize, cudaMemcpyHostToDevice));
	}

	if (opts.pVV)
	{
		CUDA_CHECK(cudaHostGetDevicePointer(&dVV, datas.VV, 0));

	}
	else
	{
		CUDA_CHECK(cudaMalloc((void **) &dVV, memsize));
		CUDA_CHECK(cudaMemcpy(dVV, datas.VV, memsize, cudaMemcpyHostToDevice));
	}

	if (!opts.isConstantPressure)
	{
		if (opts.pP)
		{
			CUDA_CHECK(cudaHostGetDevicePointer(&dP, datas.P, 0));

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

	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	if (opts.pT)
	{
		datas.pT->Unpack(dT, &stream);
	}

	if (opts.pVV)
	{
		datas.pVV->Unpack(dVV, &stream);
	}

	if (opts.pP)
	{
		datas.pP->Unpack(dP, &stream);
	}

	Calculate <<< gridSize, blockSize, 0, stream >>> (dT, dVV, dP, dVVMS, opts, dMissingValuesCount);
	
	// block until the device has completed
	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	// Retrieve result from device
	CUDA_CHECK(cudaMemcpy(&opts.missingValuesCount, dMissingValuesCount, sizeof(int), cudaMemcpyDeviceToHost));

	if (!opts.pT)
	{
		CUDA_CHECK(cudaFree(dT));
	}

	if (!opts.pVV)
	{
		CUDA_CHECK(cudaFree(dVV));
	}

	if (!opts.isConstantPressure && !opts.pP)
	{
		CUDA_CHECK(cudaFree(dP));
	}

	CUDA_CHECK(cudaFree(dMissingValuesCount));
	CUDA_CHECK(cudaStreamDestroy(stream));
}
