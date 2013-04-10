// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "cuda_helper.h"
#include "tk2tc_cuda.h"

namespace himan
{

namespace plugin
{

namespace tk2tc_cuda
{

__global__ void KernelTk2Tc(const double* __restrict__ dT, double* __restrict__ dTOut, size_t N, int* dMissingValuesCount, int* dTotalValuesCount);


} // namespace tk2tc_cuda
} // namespace plugin
} // namespace himan

__global__ void himan::plugin::tk2tc_cuda::KernelTk2Tc(const double* __restrict__ dT, double* __restrict__ dTOut, size_t N, int* dMissingValuesCount, int* dTotalValuesCount)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		atomicAdd(dTotalValuesCount, 1);
		
		if (dT[idx] == kFloatMissing)
		{
			atomicAdd(dMissingValuesCount, 1);
			dTOut[idx] = kFloatMissing;
		}
		else
		{
			dTOut[idx] = dT[idx] - 273.15;
		}
	}
}

void himan::plugin::tk2tc_cuda::DoCuda(tk2tc_cuda_options& opts)
{

	CUDA_CHECK(cudaSetDevice(opts.cudaDeviceIndex));

	// Allocate host arrays and convert input data to double

	size_t memSize = opts.N * sizeof(double);

	// Allocate device arrays

	double* dT;
	double *dTOut;	
	int *dMissingValuesCount;
	int *dTotalValuesCount;
	
	CUDA_CHECK(cudaMalloc((void **) &dT, memSize));
	CUDA_CHECK(cudaMalloc((void **) &dTOut, memSize));
	CUDA_CHECK(cudaMalloc((void **) &dMissingValuesCount, sizeof(int)));
	CUDA_CHECK(cudaMalloc((void **) &dTotalValuesCount, sizeof(int)));

	int src = 0;
	
	CUDA_CHECK(cudaMemcpy(dT, opts.TIn, memSize, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dMissingValuesCount, &src, sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dTotalValuesCount, &src, sizeof(int), cudaMemcpyHostToDevice));

	// dims

	const int blockSize = 512;
	const int gridSize = opts.N/blockSize + (opts.N%blockSize == 0?0:1);

	dim3 gridDim(gridSize);
	dim3 blockDim(blockSize);

	KernelTk2Tc <<< gridDim, blockDim >>> (dT, dTOut, opts.N, dMissingValuesCount, dTotalValuesCount);

	// block until the device has completed
	CUDA_CHECK(cudaDeviceSynchronize());

	// check if kernel execution generated an error

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	// Retrieve result from device
	CUDA_CHECK(cudaMemcpy(opts.TOut, dTOut, memSize, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(&opts.missingValuesCount, dMissingValuesCount, sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(&opts.totalValuesCount, dTotalValuesCount, sizeof(int), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(dT));
	CUDA_CHECK(cudaFree(dTOut));
	CUDA_CHECK(cudaFree(dMissingValuesCount));
	CUDA_CHECK(cudaFree(dTotalValuesCount));

}
