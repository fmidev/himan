#include "himan_common.h"
#include "numerical_functions.h"
#include "cuda_helper.h"
#include "timer.h"

#include "numerical_functions_helper.h"

#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE numerical_functions_cuda

using namespace std;
using namespace himan;

const double kEpsilon = 1e-9;

BOOST_AUTO_TEST_CASE(FILTER2DCUDA_LARGE)
{
    // Filter a plane with given filter kernel with CUDA
    himan::matrix<double> A(3007,2001,1,kFloatMissing); // input
    himan::matrix<double> B(3,3,1,kFloatMissing);       // convolution kernel
    himan::matrix<double> C(3007,2001,1,kFloatMissing); // filtered output
    himan::matrix<double> D(3007,2001,1,kFloatMissing); // reference for testing

    FilterTestSetup(A, B, D);

    const int aDimX = static_cast<int>(A.SizeX());
    const int aDimY = static_cast<int>(A.SizeY());
    const int bDimX = static_cast<int>(B.SizeX());
    const int bDimY = static_cast<int>(B.SizeY());
    const double missingValue = A.MissingValue();
        
    numerical_functions::filter_opts opts = { aDimX, aDimY, bDimX, bDimY, missingValue };
    

    const size_t aSizeBytes = A.Size() * sizeof(double);
    const size_t kernelSizeBytes = B.Size() * sizeof(double);
    const size_t filteredSizeBytes = A.Size() * sizeof(double);

    double* devInput = nullptr;
    double* devFilterKernel = nullptr;
    double* devFiltered = nullptr;

    CUDA_CHECK(cudaMalloc((void **)&devInput, aSizeBytes));
    CUDA_CHECK(cudaMalloc((void **)&devFilterKernel, kernelSizeBytes));
    CUDA_CHECK(cudaMalloc((void **)&devFiltered, filteredSizeBytes));

    CUDA_CHECK(cudaMemcpy(devInput, A.ValuesAsPOD(), aSizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devFilterKernel, B.ValuesAsPOD(), kernelSizeBytes, cudaMemcpyHostToDevice));

    const int M = A.SizeX();
    const int N = A.SizeY();

    const int blockSizeX = 32;
    const int blockSizeY = 32;

    const int gridSizeX = M / blockSizeX + (M % blockSizeX == 0 ? 0 : 1);
    const int gridSizeY = N / blockSizeY + (N % blockSizeY == 0 ? 0 : 1);
        
    const dim3 gridSize (gridSizeX, gridSizeY); 
    const dim3 blockSize (blockSizeX, blockSizeY);

    // Compute smoothened matrix with CUDA
    numerical_functions::Filter2DCuda<<<gridSize, blockSize>>>(devInput, devFilterKernel, devFiltered, opts);

    CUDA_CHECK(cudaMemcpy(C.ValuesAsPOD(), devFiltered, filteredSizeBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(devInput));
    CUDA_CHECK(cudaFree(devFilterKernel));
    CUDA_CHECK(cudaFree(devFiltered));

    // Compare results
    for(size_t i=0; i < C.Size(); ++i)
    {
        BOOST_CHECK_CLOSE(C.At(i),D.At(i),kEpsilon);
    }

    // NOTE: Don't flood the output
    #if 0
    // computed filtered matrix
    std::cout << "Matrix C computed with Filter2D:" << std::endl;
    for (size_t i=0; i < C.SizeX();++i){
        for (size_t j=0; j < C.SizeY();++j){
            std::cout << C.At(i,j,0) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Matrix D as reference case for Filter2D computation:" << std::endl; 

    for (size_t i=0; i < D.SizeX();++i){
        for (size_t j=0; j < D.SizeY();++j){
            std::cout << D.At(i,j,0) << " ";
        }
        std::cout << std::endl;
    }
    #endif
}

BOOST_AUTO_TEST_CASE(FILTER2DCUDA_SMALL)
{
    // Filter a plane with given filter kernel with CUDA
    himan::matrix<double> A(11,8,1,kFloatMissing); // input
    himan::matrix<double> B(3,3,1,kFloatMissing);  // convolution kernel
    himan::matrix<double> C(11,8,1,kFloatMissing); // filtered output
    himan::matrix<double> D(11,8,1,kFloatMissing); // reference for testing

    FilterTestSetup(A, B, D);

    const int aDimX = static_cast<int>(A.SizeX());
    const int aDimY = static_cast<int>(A.SizeY());
    const int bDimX = static_cast<int>(B.SizeX());
    const int bDimY = static_cast<int>(B.SizeY());
    const double missingValue = A.MissingValue();
        
    numerical_functions::filter_opts opts = { aDimX, aDimY, bDimX, bDimY, missingValue };
    

    const size_t aSizeBytes = A.Size() * sizeof(double);
    const size_t kernelSizeBytes = B.Size() * sizeof(double);
    const size_t filteredSizeBytes = A.Size() * sizeof(double);

    double* devInput = nullptr;
    double* devFilterKernel = nullptr;
    double* devFiltered = nullptr;

    CUDA_CHECK(cudaMalloc((void **)&devInput, aSizeBytes));
    CUDA_CHECK(cudaMalloc((void **)&devFilterKernel, kernelSizeBytes));
    CUDA_CHECK(cudaMalloc((void **)&devFiltered, filteredSizeBytes));

    CUDA_CHECK(cudaMemcpy(devInput, A.ValuesAsPOD(), aSizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devFilterKernel, B.ValuesAsPOD(), kernelSizeBytes, cudaMemcpyHostToDevice));

    const int M = A.SizeX();
    const int N = A.SizeY();

    const int blockSizeX = 32;
    const int blockSizeY = 32;

    const int gridSizeX = M / blockSizeX + (M % blockSizeX == 0 ? 0 : 1);
    const int gridSizeY = N / blockSizeY + (N % blockSizeY == 0 ? 0 : 1);
        
    const dim3 gridSize (gridSizeX, gridSizeY); 
    const dim3 blockSize (blockSizeX, blockSizeY);

    // Compute smoothened matrix with CUDA
    numerical_functions::Filter2DCuda<<<gridSize, blockSize>>>(devInput, devFilterKernel, devFiltered, opts);

    CUDA_CHECK(cudaMemcpy(C.ValuesAsPOD(), devFiltered, filteredSizeBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(devInput));
    CUDA_CHECK(cudaFree(devFilterKernel));
    CUDA_CHECK(cudaFree(devFiltered));

    // Compare results
    for(size_t i=0; i < C.Size(); ++i)
    {
        BOOST_CHECK_CLOSE(C.At(i),D.At(i),kEpsilon);
    }

    // computed filtered matrix
    std::cout << "Matrix C computed with Filter2D:" << std::endl;
    for (size_t i=0; i < C.SizeX();++i){
        for (size_t j=0; j < C.SizeY();++j){
            std::cout << C.At(i,j,0) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Matrix D as reference case for Filter2D computation:" << std::endl; 

    for (size_t i=0; i < D.SizeX();++i){
        for (size_t j=0; j < D.SizeY();++j){
            std::cout << D.At(i,j,0) << " ";
        }
        std::cout << std::endl;
    }
}

// Compare against the CPU version
BOOST_AUTO_TEST_CASE(FILTER2DCUDA_LARGE_CMP_CPU)
{
    // Filter a plane with given filter kernel with CUDA
    himan::matrix<double> A(3007,2001,1,kFloatMissing); // input
    himan::matrix<double> B(3,3,1,kFloatMissing);       // convolution kernel
    himan::matrix<double> C(3007,2001,1,kFloatMissing); // filtered output
    himan::matrix<double> D(3007,2001,1,kFloatMissing); // reference for testing

    FilterTestSetup(A, B, D);

    himan::timer CPUTimer;
    himan::timer GPUTimer;
        
    // Compute the cpu version

    CPUTimer.Start();
    himan::matrix<double> cpuResult = numerical_functions::Filter2D(A, B);
    CPUTimer.Stop();

    GPUTimer.Start();
    const int aDimX = static_cast<int>(A.SizeX());
    const int aDimY = static_cast<int>(A.SizeY());
    const int bDimX = static_cast<int>(B.SizeX());
    const int bDimY = static_cast<int>(B.SizeY());
    const double missingValue = A.MissingValue();
        
    numerical_functions::filter_opts opts = { aDimX, aDimY, bDimX, bDimY, missingValue };

    const size_t aSizeBytes = A.Size() * sizeof(double);
    const size_t kernelSizeBytes = B.Size() * sizeof(double);
    const size_t filteredSizeBytes = A.Size() * sizeof(double);

    double* devInput = nullptr;
    double* devFilterKernel = nullptr;
    double* devFiltered = nullptr;

    CUDA_CHECK(cudaMalloc((void **)&devInput, aSizeBytes));
    CUDA_CHECK(cudaMalloc((void **)&devFilterKernel, kernelSizeBytes));
    CUDA_CHECK(cudaMalloc((void **)&devFiltered, filteredSizeBytes));

    CUDA_CHECK(cudaMemcpy(devInput, A.ValuesAsPOD(), aSizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devFilterKernel, B.ValuesAsPOD(), kernelSizeBytes, cudaMemcpyHostToDevice));

    const int M = A.SizeX();
    const int N = A.SizeY();

    const int blockSizeX = 32;
    const int blockSizeY = 32;

    const int gridSizeX = M / blockSizeX + (M % blockSizeX == 0 ? 0 : 1);
    const int gridSizeY = N / blockSizeY + (N % blockSizeY == 0 ? 0 : 1);
        
    const dim3 gridSize (gridSizeX, gridSizeY); 
    const dim3 blockSize (blockSizeX, blockSizeY);

    // Compute smoothened matrix with CUDA
    numerical_functions::Filter2DCuda<<<gridSize, blockSize>>>(devInput, devFilterKernel, devFiltered, opts);

    CUDA_CHECK(cudaMemcpy(C.ValuesAsPOD(), devFiltered, filteredSizeBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(devInput));
    CUDA_CHECK(cudaFree(devFilterKernel));
    CUDA_CHECK(cudaFree(devFiltered));

    GPUTimer.Stop();

    // Compare results
    for(size_t i=0; i < C.Size(); ++i)
    {
        BOOST_CHECK_CLOSE(C.At(i),D.At(i),kEpsilon);
    }

    for(size_t i=0; i < C.Size(); ++i)
    {
        BOOST_CHECK_CLOSE(C.At(i), cpuResult.At(i),kEpsilon);
    }

    std::cout << "Filter2D(CPU) time for input matrix (" << M << "x" << N << "): " << CPUTimer.GetTime() << " ms" << std::endl;
    std::cout << "Filter2D(GPU) time for input matrix (" << M << "x" << N << "): " << GPUTimer.GetTime() << " ms" << std::endl;

    // NOTE: Don't flood the output
    #if 0
    // computed filtered matrix
    std::cout << "Matrix C computed with Filter2D:" << std::endl;
    for (size_t i=0; i < C.SizeX();++i){
        for (size_t j=0; j < C.SizeY();++j){
            std::cout << C.At(i,j,0) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Matrix D as reference case for Filter2D computation:" << std::endl; 

    for (size_t i=0; i < D.SizeX();++i){
        for (size_t j=0; j < D.SizeY();++j){
            std::cout << D.At(i,j,0) << " ";
        }
        std::cout << std::endl;
    }
    #endif
}
