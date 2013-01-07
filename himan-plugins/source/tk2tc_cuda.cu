// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#ifdef DEBUG
#include "timer_factory.h"
#endif

namespace himan
{

namespace plugin
{

namespace tk2tc_cuda
{

void doCuda(const float* Tin, float* Tout, size_t N, unsigned short deviceIndex);
void checkCUDAError(const std::string& msg);
__global__ void kernel_tk2tc(const float* Tin, float* Tout, size_t N);


} // namespace tk2tc_cuda
} // namespace plugin
} // namespace himan


const float kFloatMissing = 32700.f;

__global__ void himan::plugin::tk2tc_cuda::kernel_tk2tc(const float* Tin, float* Tout, size_t N)
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

void himan::plugin::tk2tc_cuda::doCuda(const float* Tin, float* Tout, size_t N, unsigned short deviceIndex)
{

    //cudaSetDevice(deviceIndex);
    cudaSetDevice(0); // this laptop has only one GPU

    // Allocate host arrays and convert input data to float

    size_t size = N * sizeof(float);

    // Allocate device arrays

    float* dT;
    cudaMalloc((void **) &dT, size);
    checkCUDAError("malloc dT");

    float *dTout;

    cudaMalloc((void **) &dTout, size);
    checkCUDAError("malloc dTout");

    // dims

    const int n_threads_per_block = 512;
    int n_blocks = N/n_threads_per_block + (N%n_threads_per_block == 0?0:1);

    dim3 dimGrid(n_blocks);
    dim3 dimBlock(n_threads_per_block);

#ifdef DEBUG
    timer* t = timer_factory::Instance()->GetTimer();
    t->Start();
#endif

    kernel_tk2tc <<< dimGrid, dimBlock >>> (dT, dTout, N);

    // block until the device has completed
    cudaThreadSynchronize();

    // check if kernel execution generated an error

    checkCUDAError("kernel invocation");

    // Retrieve result from device
    cudaMemcpy(Tout, dTout, size, cudaMemcpyDeviceToHost);

    checkCUDAError("memcpy");

#ifdef DEBUG
    t->Stop();

    std::cout << "cudaDebug::tpot_cuda Calculation and data transfer took " << t->GetTime() << " microseconds on GPU" << std::endl;

    delete t;
#endif

    cudaFree(dT);
    cudaFree(dTout);

}

void himan::plugin::tk2tc_cuda::checkCUDAError(const std::string& msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        std::cout << "Cuda error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}
