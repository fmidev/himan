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

__global__ void kernel_constant_pressure(const float* __restrict__ dT, float TBase, float P, float* __restrict__ TPout, size_t N);
__global__ void kernel_varying_pressure(const float* __restrict__ dT, float TBase, const float* __restrict__ dP, float PScale, float* __restrict__ TPout, size_t N);


} // namespace tpot
} // namespace plugin
} // namespace himan

__global__ void himan::plugin::tpot_cuda::kernel_constant_pressure(const float* __restrict__ dT,
																	float TBase, float P,
																	float* __restrict__ TPout, size_t N)
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

__global__ void himan::plugin::tpot_cuda::kernel_varying_pressure(const float* __restrict__ dT, float TBase,
																	const float* __restrict__ dP, float PScale,
																	float* __restrict__ TPout, size_t N)
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


void himan::plugin::tpot_cuda::DoCuda(const float* Tin, float TBase, const float* Pin, float PScale, float* TPout, size_t N, float PConst, unsigned short deviceIndex)
{

	cudaSetDevice(deviceIndex);
	CheckCudaError("deviceset");

	// Allocate host arrays and convert input data to float

	size_t memSize = N * sizeof(float);

	bool isConstantPressure = (Pin == 0 && PConst > 0);

	// Allocate device arrays

	float* dT;
	cudaMalloc((void **) &dT, memSize);
	CheckCudaError("malloc dT");

	float* dP;

	if (!isConstantPressure)
	{
		cudaMalloc((void **) &dP, memSize);
		CheckCudaError("malloc dP");
	}

	float *dTP;

	cudaMalloc((void **) &dTP, memSize);
	CheckCudaError("malloc dTP");

	cudaMemcpy(dT, Tin, memSize, cudaMemcpyHostToDevice);
	CheckCudaError("memcpy Tin");

	if (!isConstantPressure)
	{
		cudaMemcpy(dP, Pin, memSize, cudaMemcpyHostToDevice);
		CheckCudaError("memcpy Pin");
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
	cudaDeviceSynchronize();

	// check if kernel execution generated an error

	CheckCudaError("kernel invocation");

	// Retrieve result from device
	cudaMemcpy(TPout, dTP, memSize, cudaMemcpyDeviceToHost);

	CheckCudaError("memcpy dTP");

	cudaFree(dT);

	if (!isConstantPressure)
	{
		cudaFree(dP);
	}

	cudaFree(dTP);

}
