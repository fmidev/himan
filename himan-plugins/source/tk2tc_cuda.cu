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

__global__ void kernel_tk2tc(const float* __restrict__ Tin, float* __restrict__ Tout, size_t N);


} // namespace tk2tc_cuda
} // namespace plugin
} // namespace himan

__global__ void himan::plugin::tk2tc_cuda::kernel_tk2tc(const float* __restrict__ Tin, float* __restrict__ Tout, size_t N)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {

        if (Tin[idx] == kFloatMissing)
        {
            Tout[idx] = kFloatMissing;
        }
        else
        {
            Tout[idx] = Tin[idx] - 273.15f;
        }
    }
}

void himan::plugin::tk2tc_cuda::DoCuda(const float* Tin, float* Tout, size_t N, unsigned short deviceIndex)
{

    cudaSetDevice(deviceIndex); // this laptop has only one GPU
    CheckCudaError("deviceset");

    // Allocate host arrays and convert input data to float

    size_t size = N * sizeof(float);

    // Allocate device arrays

    float* dT;
    cudaMalloc((void **) &dT, size);
    CheckCudaError("malloc dT");

    float *dTout;

    cudaMalloc((void **) &dTout, size);
    CheckCudaError("malloc dTout");

    cudaMemcpy(dT, Tin, size, cudaMemcpyHostToDevice);

    CheckCudaError("memcpy");

    // dims

    const int n_threads_per_block = 512;
    int n_blocks = N/n_threads_per_block + (N%n_threads_per_block == 0?0:1);

    dim3 dimGrid(n_blocks);
    dim3 dimBlock(n_threads_per_block);

#ifdef DEBUG
    std::cout << "cudaDebug::tpot_cuda blocksize: " << n_threads_per_block << " gridsize: " << n_blocks << std::endl;
#endif

    kernel_tk2tc <<< dimGrid, dimBlock >>> (dT, dTout, N);

    // block until the device has completed
    cudaThreadSynchronize();

    // check if kernel execution generated an error

    CheckCudaError("kernel invocation");

    // Retrieve result from device
    cudaMemcpy(Tout, dTout, size, cudaMemcpyDeviceToHost);

    CheckCudaError("memcpy");

    cudaFree(dT);
    cudaFree(dTout);

}
