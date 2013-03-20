// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "cuda_extern.h"
#include "cuda_helper.h"

namespace himan
{

namespace plugin
{

namespace vvms_cuda
{

__global__ void kernel_constant_pressure(const float* __restrict__ dT, float TBase, float P, const float* __restrict__ dVVPas, float* __restrict__ VVout, size_t N);
__global__ void kernel_varying_pressure(const float* __restrict__ dT, float TBase, const float* __restrict__ dP, float PScale, const float* __restrict__ dVVPas, float* __restrict__ VVout, size_t N);


} // namespace tpot
} // namespace plugin
} // namespace himan

__global__ void himan::plugin::vvms_cuda::kernel_constant_pressure(const float* __restrict__ dT, float TBase, float P, const float* __restrict__ dVVPas, float* __restrict__ VVout, size_t N)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{

		if (dT[idx] == kFloatMissing || P == kFloatMissing || dVVPas[idx] == kFloatMissing)
		{
			VVout[idx] = kFloatMissing;
		}
		else
		{
			VVout[idx] = 287.f * -dVVPas[idx] * (TBase + dT[idx]) / (9.81f * P);
		}
	}
}

__global__ void himan::plugin::vvms_cuda::kernel_varying_pressure(const float* dT, float TBase, const float* __restrict__ dP, float PScale, const float* __restrict__ dVVPas, float* __restrict__ VVout, size_t N)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{

		if (dT[idx] == kFloatMissing || dP[idx] == kFloatMissing || dVVPas[idx] == kFloatMissing)
		{
			VVout[idx] = kFloatMissing;
		}
		else
		{
			VVout[idx] = 287.f * -dVVPas[idx] * (TBase + dT[idx]) / (9.81f * dP[idx] * PScale);
		}
	}
}


void himan::plugin::vvms_cuda::DoCuda(const float* Tin, float TBase, const float* Pin, float PScale, const float* VVin, float* VVout, size_t N, float PConst, unsigned short deviceIndex)
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

	float *dVVPas;

	cudaMalloc((void **) &dVVPas, memSize);
	CheckCudaError("malloc dVVPas");

	float *dVVout;

	cudaMalloc((void **) &dVVout, memSize);
	CheckCudaError("malloc dVVout");

	cudaMemcpy(dT, Tin, memSize, cudaMemcpyHostToDevice);
	CheckCudaError("memcpy dT");

	if (!isConstantPressure)
	{
		cudaMemcpy(dP, Pin, memSize, cudaMemcpyHostToDevice);
		CheckCudaError("memcpy dP");
	}

	cudaMemcpy(dVVPas, VVin, memSize, cudaMemcpyHostToDevice);
	CheckCudaError("memcpy dVVPas");

	// dims

	const int blockSize = 512;
	const int gridSize = N/blockSize + (N%blockSize == 0?0:1);

	dim3 gridDim(gridSize);
	dim3 blockDim(blockSize);

	if (isConstantPressure)
	{
		kernel_constant_pressure <<< gridDim, blockDim >>> (dT, TBase, PConst, dVVPas, dVVout, N);
	}
	else
	{
		kernel_varying_pressure <<< gridDim, blockDim >>> (dT, TBase, dP, PScale, dVVPas, dVVout, N);
	}

	// block until the device has completed
	cudaDeviceSynchronize();

	// check if kernel execution generated an error

	CheckCudaError("kernel invocation");

	// Retrieve result from device
	cudaMemcpy(VVout, dVVout, memSize, cudaMemcpyDeviceToHost);

	CheckCudaError("memcpy");

	cudaFree(dT);

	if (!isConstantPressure)
	{
		cudaFree(dP);
	}

	cudaFree(dVVPas);
	cudaFree(dVVout);

}
