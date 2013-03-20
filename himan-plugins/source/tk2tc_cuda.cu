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

namespace tk2tc_cuda
{

__global__ void kernel_tk2tc(const float* __restrict__ dT, float* __restrict__ dTout, size_t N);


} // namespace tk2tc_cuda
} // namespace plugin
} // namespace himan

__global__ void himan::plugin::tk2tc_cuda::kernel_tk2tc(const float* __restrict__ dT, float* __restrict__ dTout, size_t N)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{

		if (dT[idx] == kFloatMissing)
		{
			dTout[idx] = kFloatMissing;
		}
		else
		{
			dTout[idx] = dT[idx] - 273.15f;
		}
	}
}

void himan::plugin::tk2tc_cuda::DoCuda(const float* Tin, float* Tout, size_t N, unsigned short deviceIndex)
{

	cudaSetDevice(deviceIndex); // this laptop has only one GPU
	CheckCudaError("deviceset");

	// Allocate host arrays and convert input data to float

	size_t memSize = N * sizeof(float);

	// Allocate device arrays

	float* dT;
	cudaMalloc((void **) &dT, memSize);
	CheckCudaError("malloc dT");

	float *dTout;

	cudaMalloc((void **) &dTout, memSize);
	CheckCudaError("malloc dTout");

	cudaMemcpy(dT, Tin, memSize, cudaMemcpyHostToDevice);

	CheckCudaError("memcpy");

	// dims

	const int blockSize = 512;
	const int gridSize = N/blockSize + (N%blockSize == 0?0:1);

	dim3 gridDim(gridSize);
	dim3 blockDim(blockSize);

	kernel_tk2tc <<< gridDim, blockDim >>> (dT, dTout, N);

	// block until the device has completed
	cudaDeviceSynchronize();

	// check if kernel execution generated an error

	CheckCudaError("kernel invocation");

	// Retrieve result from device
	cudaMemcpy(Tout, dTout, memSize, cudaMemcpyDeviceToHost);

	CheckCudaError("memcpy");

	cudaFree(dT);
	cudaFree(dTout);

}
