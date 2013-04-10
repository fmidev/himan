// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "cuda_helper.h"
#include "cuda_extern.h"

namespace himan
{

namespace plugin
{

namespace tpot_cuda
{

__global__ void kernel_constant_pressure(const double* __restrict__ dT, double TBase, double P, double* __restrict__ TPout, size_t N);
__global__ void kernel_varying_pressure(const double* __restrict__ dT, double TBase, const double* __restrict__ dP, double PScale, double* __restrict__ TPout, size_t N);


} // namespace tpot
} // namespace plugin
} // namespace himan

__global__ void himan::plugin::tpot_cuda::kernel_constant_pressure(const double* __restrict__ dT,
																	double TBase, double P,
																	double* __restrict__ TPout, size_t N)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{

		if (dT[idx] == kFloatMissing || P == kFloatMissing)
		{
			TPout[idx] = kFloatMissing;
		}
		else
		{
			TPout[idx] = (TBase + dT[idx]) * powf((1000.f / P), 0.286f);
		}
	}
}

__global__ void himan::plugin::tpot_cuda::kernel_varying_pressure(const double* __restrict__ dT, double TBase,
																	const double* __restrict__ dP, double PScale,
																	double* __restrict__ TPout, size_t N)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{

		if (dT[idx] == kFloatMissing || dP[idx] == kFloatMissing)
		{
			TPout[idx] = kFloatMissing;
		}
		else
		{
			TPout[idx] = (TBase + dT[idx]) * powf((1000.f / (PScale * dP[idx])), 0.286f);
		}
	}
}


void himan::plugin::tpot_cuda::DoCuda(const double* Tin, double TBase, const double* Pin, double PScale, double* TPout, size_t N, double PConst, unsigned short deviceIndex)
{

	CUDA_CHECK(cudaSetDevice(deviceIndex));

	size_t memSize = N * sizeof(double);

	bool isConstantPressure = (Pin == 0 && PConst > 0);

	// Allocate device arrays

	double* dT;
	double* dP;
	double* dTP;
	
	CUDA_CHECK(cudaMalloc((void **) &dT, memSize));
	CUDA_CHECK(cudaMalloc((void **) &dTP, memSize));

	if (!isConstantPressure)
	{
		CUDA_CHECK(cudaMalloc((void **) &dP, memSize));
	}

	CUDA_CHECK(cudaMemcpy(dT, Tin, memSize, cudaMemcpyHostToDevice));

	if (!isConstantPressure)
	{
		CUDA_CHECK(cudaMemcpy(dP, Pin, memSize, cudaMemcpyHostToDevice));
	}

	// dims

	const int blockSize = 512;
	const int gridSize = N/blockSize + (N%blockSize == 0?0:1);

	dim3 gridDim(gridSize);
	dim3 blockDim(blockSize);

	if (isConstantPressure)
	{
		kernel_constant_pressure <<< gridDim, blockDim >>> (dT, TBase, PConst, dTP, N);
	}
	else
	{
		kernel_varying_pressure <<< gridDim, blockDim >>> (dT, TBase, dP, PScale, dTP, N);
	}

	// block until the device has completed
	CUDA_CHECK(cudaDeviceSynchronize());

	// check if kernel execution generated an error

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	// Retrieve result from device
	CUDA_CHECK(cudaMemcpy(TPout, dTP, memSize, cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(dT));

	if (!isConstantPressure)
	{
		CUDA_CHECK(cudaFree(dP));
	}

	CUDA_CHECK(cudaFree(dTP));

}
