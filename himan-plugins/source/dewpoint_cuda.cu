// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "cuda_helper.h"
#include "cuda_extern.h"

const double RW = 461.5f; // Vesihoyryn kaasuvakio (J / K kg)
const double L = 2.5e6f; // Veden hoyrystymislampo (J / kg)
const double RW_div_L = RW / L;

namespace himan
{

namespace plugin
{

namespace dewpoint_cuda
{

__global__ void kernel_dewpoint(const double* __restrict__ dT, double TBase, const double* __restrict__ dRH, double* __restrict__ dDP, size_t N);


} // namespace dewpoint
} // namespace plugin
} // namespace himan

__global__ void himan::plugin::dewpoint_cuda::kernel_dewpoint(const double* __restrict__ dT, double TBase,
																const double* __restrict__ dRH,
																double* __restrict__ dDP, size_t N)
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

void himan::plugin::dewpoint_cuda::DoCuda(const double* Tin, double TBase, const double* RHin, double* DPout, size_t N, unsigned short deviceIndex)
{

	CUDA_CHECK(cudaSetDevice(deviceIndex));

	size_t memSize = N * sizeof(double);

	// Allocate device arrays

	double* dT;
	double* dRH;
	double* dDP;

	CUDA_CHECK(cudaMalloc((void **) &dT, memSize));
	CUDA_CHECK(cudaMalloc((void **) &dRH, memSize));
	CUDA_CHECK(cudaMalloc((void **) &dDP, memSize));
	
	CUDA_CHECK(cudaMemcpy(dT, Tin, memSize, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dRH, RHin, memSize, cudaMemcpyHostToDevice));
	
	// dims

	const int blockSize = 512;
	const int gridSize = N/blockSize + (N%blockSize == 0?0:1);

	dim3 gridDim(gridSize);
	dim3 blockDim(blockSize);

	kernel_dewpoint <<< gridDim, blockDim >>> (dT, TBase, dRH, dDP, N);

	// block until the device has completed
	CUDA_CHECK(cudaDeviceSynchronize());

	// check if kernel execution generated an error

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	// Retrieve result from device
	CUDA_CHECK(cudaMemcpy(DPout, dDP, memSize, cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(dT));
	CUDA_CHECK(cudaFree(dRH));
	CUDA_CHECK(cudaFree(dDP));

}
