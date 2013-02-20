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

__global__ void kernel_constant_pressure(float* Tin, float TBase, float P, float* TPout, size_t N);
__global__ void kernel_varying_pressure(float* Tin, float TBase, float* Pin, float PScale, float* TPout, size_t N);


} // namespace tpot
} // namespace plugin
} // namespace himan

__global__ void himan::plugin::tpot_cuda::kernel_constant_pressure(float* Tin, float TBase, float P, float* TPout, size_t N)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {

        if (Tin[idx] == kFloatMissing || P == kFloatMissing)
        {
            TPout[idx] = kFloatMissing;
        }
        else
        {
            TPout[idx] = (TBase + Tin[idx]) * powf((1000 / P), 0.286f);
        }
    }
}

__global__ void himan::plugin::tpot_cuda::kernel_varying_pressure(float* Tin, float TBase, float* Pin, float PScale, float* TPout, size_t N)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {

        if (Tin[idx] == kFloatMissing || Pin[idx] == kFloatMissing)
        {
            TPout[idx] = kFloatMissing;
        }
        else
        {
            TPout[idx] = (TBase + Tin[idx]) * powf((1000 / (PScale * Pin[idx])), 0.286f);
        }
    }
}


void himan::plugin::tpot_cuda::DoCuda(const float* Tin, float TBase, const float* Pin, float PScale, float* TPout, size_t N, float
PConst, unsigned short deviceIndex)
{

    cudaSetDevice(deviceIndex);
    CheckCudaError("deviceset");

#ifdef CUDA_STREAMS
    cudaStream_t stream;

    cudaStreamCreate(&stream);

    cudaError_t err;t

#endif

    // Allocate host arrays and convert input data to float

    size_t size = N * sizeof(float);

    bool isConstantPressure = (Pin == 0 && PConst > 0);

    // Allocate device arrays

    float* dT;
    cudaMalloc((void **) &dT, size);
    CheckCudaError("malloc dT");

    float* dP;

    if (!isConstantPressure)
    {
        cudaMalloc((void **) &dP, size);
        CheckCudaError("malloc dP");
    }

#ifdef CUDA_STREAMS
    float *TPpinned;
    CheckCudaErrors(cudaMallocHost((void **)&TPpinned, size));
#endif

    float *dTP;

    cudaMalloc((void **) &dTP, size);
    CheckCudaError("malloc dTP");

#ifdef CUDA_STREAMS
    cudaMemcpyAsync(dT, Tin, size, cudaMemcpyHostToDevice, stream);
#else
    cudaMemcpy(dT, Tin, size, cudaMemcpyHostToDevice);
    CheckCudaError("memcpy Tin");
#endif


    if (!isConstantPressure)
    {
#ifdef CUDA_STREAMS
        cudaMemcpyAsync(dP, Pin, size, cudaMemcpyHostToDevice, stream);
#else
        cudaMemcpy(dP, Pin, size, cudaMemcpyHostToDevice);
        CheckCudaError("memcpy Pin");
#endif

    }

#ifdef CUDA_STREAMS
    cudaMemcpyAsync(dTP, TPout, size, cudaMemcpyHostToDevice, stream);
#else
    cudaMemcpy(dTP, TPout, size, cudaMemcpyHostToDevice);
    CheckCudaError("memcpy TPout");
#endif

    // dims

    const int n_threads_per_block = 512;
    int n_blocks = N/n_threads_per_block + (N%n_threads_per_block == 0?0:1);

    dim3 dimGrid(n_blocks,1);
    dim3 dimBlock(n_threads_per_block, 1);

    if (isConstantPressure)
    {
#ifdef CUDA_STREAMS
        kernel_constant_pressure <<< dimGrid, dimBlock, 0, stream >>> (dT, TBase, PConst, dTP, N);
#else
        kernel_constant_pressure <<< dimGrid, dimBlock >>> (dT, TBase, PConst, dTP, N);
#endif
    }
    else
    {
#ifdef CUDA_STREAMS
        kernel_varying_pressure <<< dimGrid, dimBlock, 0, stream >>> (dT, TBase, dP, PScale, dTP, N);
#else
        kernel_varying_pressure <<< dimGrid, dimBlock >>> (dT, TBase, dP, PScale, dTP, N);
#endif
    }

    // block until the device has completed

#ifdef CUDA_STREAMS
    cudaMemcpyAsync(TPpinned, dTP, size, cudaMemcpyDeviceToHost, stream);

    if ((err = cudaStreamSynchronize(stream)) != cudaSuccess)
    {
        std::cout << "Cuda error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
#else

    cudaThreadSynchronize();
    // check if kernel execution generated an error

    CheckCudaError("kernel invocation");

    // Retrieve result from device
    cudaMemcpy(TPout, dTP, size, cudaMemcpyDeviceToHost);

    CheckCudaError("memcpy dTP");

#endif

    cudaFree(dT);

    if (!isConstantPressure)
    {
        cudaFree(dP);
    }

    cudaFree(dTP);

#ifdef CUDA_STREAMS
    TPout = TPpinned;

    cudaFreeHost(TPpinned);
    cudaStreamDestroy(stream);
#endif
}
