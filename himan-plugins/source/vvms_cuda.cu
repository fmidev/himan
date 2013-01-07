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

namespace vvms_cuda
{

void doCuda(const float* Tin, float TBase, const float* Pin, float PScale, const float* VVin, float* VVout, size_t N, float PConst, unsigned short deviceIndex);
void checkCUDAError(const std::string& msg);
__global__ void kernel_constant_pressure(float* Tin, float TBase, float P, float* VVin, float* VVout, size_t N);
__global__ void kernel_varying_pressure(float* Tin, float TBase, float* Pin, float PScale, float* VVin, float* VVout, size_t N);


} // namespace tpot
} // namespace plugin
} // namespace himan


const float kFloatMissing = 32700.f;

__global__ void himan::plugin::vvms_cuda::kernel_constant_pressure(float* Tin, float TBase, float P, float* VVin, float* VVout, size_t N)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {

        if (Tin[idx] == kFloatMissing || P == kFloatMissing || VVin[idx] == kFloatMissing)
        {
            VVout[idx] = kFloatMissing;
        }
        else
        {
            //double VVms = 287 * -VV * (T + TBase) / (9.81 * (P * PScale));

            VVout[idx] = 287.f * -VVin[idx] * (TBase + Tin[idx]) / (9.81f * P);
        }
    }
}

__global__ void himan::plugin::vvms_cuda::kernel_varying_pressure(float* Tin, float TBase, float* Pin, float PScale, float* VVin, float* VVout, size_t N)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {

        if (Tin[idx] == kFloatMissing || Pin[idx] == kFloatMissing || VVin[idx] == kFloatMissing)
        {
            VVout[idx] = kFloatMissing;
        }
        else
        {
            //double VVms = 287 * -VV * (T + TBase) / (9.81 * (P * PScale));

            VVout[idx] = 287.f * -VVin[idx] * (TBase + Tin[idx]) / (9.81f * Pin[idx] * PScale);
        }
    }
}


void himan::plugin::vvms_cuda::doCuda(const float* Tin, float TBase, const float* Pin, float PScale, const float* VVin, float* VVout, size_t N, float PConst, unsigned short deviceIndex)
{

    //cudaSetDevice(deviceIndex);
    cudaSetDevice(0); // this laptop has only one GPU

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

    float *dVVin;

    cudaMalloc((void **) &dVVin, size);
    checkCUDAError("malloc dVVin");

    float *dVVout;

    cudaMalloc((void **) &dVVout, size);
    checkCUDAError("malloc dVVout");

    cudaMemcpy(dT, Tin, size, cudaMemcpyHostToDevice);
    checkCUDAError("memcpy Tin");

    if (!isConstantPressure)
    {
        cudaMemcpy(dP, Pin, size, cudaMemcpyHostToDevice);
        checkCUDAError("memcpy Pin");
    }

    cudaMemcpy(dVVin, VVin, size, cudaMemcpyHostToDevice);
    checkCUDAError("memcpy VVin");

    cudaMemcpy(dVVout, VVout, size, cudaMemcpyHostToDevice);
    checkCUDAError("memcpy VVout");

    // dims

    const int n_threads_per_block = 512;
    int n_blocks = N/n_threads_per_block + (N%n_threads_per_block == 0?0:1);

    dim3 dimGrid(n_blocks);
    dim3 dimBlock(n_threads_per_block);

#ifdef DEBUG
    timer* t = timer_factory::Instance()->GetTimer();
    t->Start();
#endif

    if (isConstantPressure)
    {
        kernel_constant_pressure <<< dimGrid, dimBlock >>> (dT, TBase, PConst, dVVin, dVVout, N);
    }
    else
    {
        kernel_varying_pressure <<< dimGrid, dimBlock >>> (dT, TBase, dP, PScale, dVVin, dVVout, N);
    }

    // block until the device has completed
    cudaThreadSynchronize();

    // check if kernel execution generated an error

    checkCUDAError("kernel invocation");

    // Retrieve result from device
    cudaMemcpy(VVout, dVVout, size, cudaMemcpyDeviceToHost);

    checkCUDAError("memcpy");

#ifdef DEBUG
    t->Stop();

    std::cout << "cudaDebug::tpot_cuda Calculation and data transfer took " << t->GetTime() << " microseconds on GPU" << std::endl;

    delete t;
#endif

    cudaFree(dT);

    if (!isConstantPressure)
    {
        cudaFree(dP);
    }
    cudaFree(dVVin);
    cudaFree(dVVout);

}

void himan::plugin::vvms_cuda::checkCUDAError(const std::string& msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        std::cout << "Cuda error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}
