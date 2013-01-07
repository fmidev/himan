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

//#define CUDA_CHECK(a,msg) checkCUDAError(#a,__FILE__,__LINE__,a,msg)
//#define CUDA_STREAMS

namespace himan
{

namespace plugin
{

namespace tpot_cuda
{

void doCuda(const float* Tin, float TBase, const float* Pin, float PScale, float* TPout, size_t N, float PConst, unsigned short index);
void checkCUDAError(const std::string& msg);
__global__ void kernel_constant_pressure(float* Tin, float TBase, float P, float* TPout, size_t N);
__global__ void kernel_varying_pressure(float* Tin, float TBase, float* Pin, float PScale, float* TPout, size_t N);


} // namespace tpot
} // namespace plugin
} // namespace himan


const float kFloatMissing = 32700.f;

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


void himan::plugin::tpot_cuda::doCuda(const float* Tin, float TBase, const float* Pin, float PScale, float* TPout, size_t N, float PConst, unsigned short index)
{

    //cudaSetDevice(deviceIndex);
    //cudaSetDevice(0); // this laptop has only one GPU

#ifdef CUDA_STREAMS
    cudaStream_t stream;

    cudaStreamCreate(&stream);

    cudaError_t err;

#endif

    std::cout << "P " << PConst << std::endl;
    // Allocate host arrays and convert input data to float

    size_t size = N * sizeof(float);

    bool isConstantPressure = (Pin == 0 && PConst > 0);

    // Allocate device arrays

    float* dT;
    cudaMalloc((void **) &dT, size);
    checkCUDAError("malloc dT");

    float* dP;

    if (!isConstantPressure)
    {
        cudaMalloc((void **) &dP, size);
        checkCUDAError("malloc dP");
    }

#ifdef CUDA_STREAMS
    float *TPpinned;
    checkCudaErrors(cudaMallocHost((void **)&TPpinned, size));
#endif

    float *dTP;

    cudaMalloc((void **) &dTP, size);
    checkCUDAError("malloc dTP");

#ifdef CUDA_STREAMS
    cudaMemcpyAsync(dT, Tin, size, cudaMemcpyHostToDevice, stream);
#else
    cudaMemcpy(dT, Tin, size, cudaMemcpyHostToDevice);
    checkCUDAError("memcpy Tin");
#endif


    if (!isConstantPressure)
    {
#ifdef CUDA_STREAMS
        cudaMemcpyAsync(dP, Pin, size, cudaMemcpyHostToDevice, stream);
#else
        cudaMemcpy(dP, Pin, size, cudaMemcpyHostToDevice);
        checkCUDAError("memcpy Pin");
#endif

    }

#ifdef CUDA_STREAMS
    cudaMemcpyAsync(dTP, TPout, size, cudaMemcpyHostToDevice, stream);
#else
    cudaMemcpy(dTP, TPout, size, cudaMemcpyHostToDevice);
    checkCUDAError("memcpy TPout");
#endif

    // dims

    const int n_threads_per_block = 512;
    int n_blocks = N/n_threads_per_block + (N%n_threads_per_block == 0?0:1);

    dim3 dimGrid(n_blocks,1);
    dim3 dimBlock(n_threads_per_block, 1);

#ifdef DEBUG
    timer* t = timer_factory::Instance()->GetTimer();
    t->Start();
#endif

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

#ifdef DEBUG
    t->Stop();

    std::cout << "cudaDebug::tpot_cuda Kernel execution took took " << t->GetTime() << " microseconds" << std::endl;

    delete t;
#endif

    checkCUDAError("kernel invocation");

    // Retrieve result from device
    cudaMemcpy(TPout, dTP, size, cudaMemcpyDeviceToHost);

    checkCUDAError("memcpy dTP");

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

void himan::plugin::tpot_cuda::checkCUDAError(const std::string& msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        std::cout << "Cuda error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}
