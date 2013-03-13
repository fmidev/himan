// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "cuda_extern.h"
#include "cuda_helper.h"

const float kRadToDeg = 57.295779513082; // 180 / PI

namespace himan
{

namespace plugin
{

namespace windvector_cuda
{

__global__ void kernel_windvector(float* __restrict__ Uin, float* __restrict__ Vin, float* __restrict__ dataOut, size_t N,  bool doVector);


} // namespace tpot
} // namespace plugin
} // namespace himan

__global__ void himan::plugin::windvector_cuda::kernel_windvector(float* __restrict__ Uin, float* __restrict__ Vin, float* __restrict__ dataOut, size_t N,  bool doVector)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
    if (idx < N)
    {

		if (Uin[idx] == kFloatMissing ||Vin[idx] == kFloatMissing)
		{
			dataOut[idx] = kFloatMissing;
			dataOut[idx+N] = kFloatMissing;

			if (doVector)
			{
				dataOut[idx+2*N] = kFloatMissing;
			}
		}
		else
		{
			float U = Uin[idx], V = Vin[idx];

			float speed = sqrt(U*U + V*V);
			float dir = 0;

			if (speed > 0)
			{
				dir = round(kRadToDeg * atan2(U,V) + 180); // Rounding dir
			}

			dataOut[idx] = speed;
			dataOut[idx+N] = dir;

			if (doVector)
			{
				float vector = round(U/10) + 100 * round(V);

				dataOut[idx+2*N] = vector;
			}

		}
	}
}

void himan::plugin::windvector_cuda::DoCuda(const float* Uin, const float* Vin, float* dataOut, size_t N, bool doVector, unsigned short deviceIndex)
{

    cudaSetDevice(deviceIndex);
    CheckCudaError("deviceset");

    // Allocate host arrays and convert input data to float

    size_t size = N * sizeof(float);

    // Allocate device arrays

    float* dU;
    cudaMalloc((void **) &dU, size);
    CheckCudaError("malloc dU");

    float* dV;

    cudaMalloc((void **) &dV, size);
    CheckCudaError("malloc dV");
    
    float *dDataOut;

	if (doVector)
	{
		cudaMalloc((void **) &dDataOut, 3*size);
	}
	else
	{
		cudaMalloc((void **) &dDataOut, 2*size);
	}
	
    CheckCudaError("malloc dDataOut");

    cudaMemcpy(dU, Uin, size, cudaMemcpyHostToDevice);
    CheckCudaError("memcpy Uin");

    cudaMemcpy(dV, Vin, size, cudaMemcpyHostToDevice);
    CheckCudaError("memcpy Vin");

    // dims

    const int n_threads_per_block = 512;
    int n_blocks = N/n_threads_per_block + (N%n_threads_per_block == 0?0:1);

    dim3 dimGrid(n_blocks);
    dim3 dimBlock(n_threads_per_block);

    kernel_windvector <<< dimGrid, dimBlock >>> (dU, dV, dDataOut, N, doVector);
 
    // block until the device has completed
    cudaThreadSynchronize();

    // check if kernel execution generated an error

    CheckCudaError("kernel invocation");

    // Retrieve result from device
    cudaMemcpy(dataOut, dDataOut, size, cudaMemcpyDeviceToHost);

    CheckCudaError("memcpy");

    cudaFree(dU);
    cudaFree(dV);
    cudaFree(dDataOut);

}
