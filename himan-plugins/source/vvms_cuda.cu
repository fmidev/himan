// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "cuda_extern.h"
#include "cuda_helper.h"

namespace himan
{

namespace plugin
{

namespace vvms_cuda
{

__global__ void kernel_constant_pressure(float* Tin, float TBase, float P, float* VVin, float* VVout, size_t N);
__global__ void kernel_varying_pressure(float* Tin, float TBase, float* Pin, float PScale, float* VVin, float* VVout, size_t N);


} // namespace tpot
} // namespace plugin
} // namespace himan

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


void himan::plugin::vvms_cuda::DoCuda(const float* Tin, float TBase, const float* Pin, float PScale, const float* VVin, float* VVout, size_t N, float PConst, unsigned short deviceIndex)
{

    cudaSetDevice(deviceIndex);
    CheckCudaError("deviceset");

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

    float *dVVin;

    cudaMalloc((void **) &dVVin, size);
    CheckCudaError("malloc dVVin");

    float *dVVout;

    cudaMalloc((void **) &dVVout, size);
    CheckCudaError("malloc dVVout");

    cudaMemcpy(dT, Tin, size, cudaMemcpyHostToDevice);
    CheckCudaError("memcpy Tin");

    if (!isConstantPressure)
    {
        cudaMemcpy(dP, Pin, size, cudaMemcpyHostToDevice);
        CheckCudaError("memcpy Pin");
    }

    cudaMemcpy(dVVin, VVin, size, cudaMemcpyHostToDevice);
    CheckCudaError("memcpy VVin");

    cudaMemcpy(dVVout, VVout, size, cudaMemcpyHostToDevice);
    CheckCudaError("memcpy VVout");

    // dims

    const int n_threads_per_block = 512;
    int n_blocks = N/n_threads_per_block + (N%n_threads_per_block == 0?0:1);

    dim3 dimGrid(n_blocks);
    dim3 dimBlock(n_threads_per_block);

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

    CheckCudaError("kernel invocation");

    // Retrieve result from device
    cudaMemcpy(VVout, dVVout, size, cudaMemcpyDeviceToHost);

    CheckCudaError("memcpy");

    cudaFree(dT);

    if (!isConstantPressure)
    {
        cudaFree(dP);
    }
    cudaFree(dVVin);
    cudaFree(dVVout);

}
