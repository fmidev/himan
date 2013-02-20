#include <cuda_runtime.h>

const float kFloatMissing = 32700.f;

void CheckCudaError(const std::string& msg);

void CheckCudaError(const std::string& msg)
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        std::cerr << "Cuda error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

