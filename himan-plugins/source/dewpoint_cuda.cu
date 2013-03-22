// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "cuda_helper.h"
#include "cuda_extern.h"

const float RW = 461.5f; // Vesihoyryn kaasuvakio (J / K kg)
const float L = 2.5e6f; // Veden hoyrystymislampo (J / kg)
const float RW_div_L = RW / L;

namespace himan
{

namespace plugin
{

namespace dewpoint_cuda
{

__global__ void kernel_dewpoint(const float* __restrict__ dT, float TBase, const float* __restrict__ dRH, float* __restrict__ dDP, size_t N);


} // namespace dewpoint
} // namespace plugin
} // namespace himan

__global__ void himan::plugin::dewpoint_cuda::kernel_dewpoint(const float* __restrict__ dT, float TBase,
																const float* __restrict__ dRH,
																float* __restrict__ dDP, size_t N)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{

		if (dT[idx] == kFloatMissing || dRH[idx] == kFloatMissing)
		{
			dDP[idx] = kFloatMissing;
		}
		else
		{
			dDP[idx] = ((dT[idx]+TBase) / (1.f - ((dT[idx]+TBase) * logf(dRH[idx]) * (RW_div_L)))) - 273.15f + TBase;
		}
	}
}

void himan::plugin::dewpoint_cuda::DoCuda(const float* Tin, float TBase, const float* RHin, float* DPout, size_t N, unsigned short deviceIndex)
{

	cudaSetDevice(deviceIndex);
	CheckCudaError("deviceset");

	// Allocate host arrays and convert input data to float

	size_t memSize = N * sizeof(float);

	// Allocate device arrays

	float* dT;
	cudaMalloc((void **) &dT, memSize);
	CheckCudaError("malloc dT");

	float *dRH;

	cudaMalloc((void **) &dRH, memSize);
	CheckCudaError("malloc dRH");

	float *dDP;

	cudaMalloc((void **) &dDP, memSize);
	CheckCudaError("malloc dDP");

	cudaMemcpy(dT, Tin, memSize, cudaMemcpyHostToDevice);
	CheckCudaError("memcpy Tin");

	cudaMemcpy(dRH, RHin, memSize, cudaMemcpyHostToDevice);
	CheckCudaError("memcpy RHin");

	// dims

	const int blockSize = 512;
	const int gridSize = N/blockSize + (N%blockSize == 0?0:1);

	dim3 gridDim(gridSize);
	dim3 blockDim(blockSize);

	kernel_dewpoint <<< gridDim, blockDim >>> (dT, TBase, dRH, dDP, N);

	// block until the device has completed
	cudaDeviceSynchronize();

	// check if kernel execution generated an error

	CheckCudaError("kernel invocation");

	// Retrieve result from device
	cudaMemcpy(DPout, dDP, memSize, cudaMemcpyDeviceToHost);

	CheckCudaError("memcpy dDP");

	cudaFree(dT);
	cudaFree(dRH);

}
