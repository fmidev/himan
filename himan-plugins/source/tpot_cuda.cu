// System includes
#include <iostream>
#include <string>
#include <math.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

const float kFloatMissing = 32700.f;

timespec start_ts;
timespec stop_ts;

void checkCUDAError(const std::string& msg);
void StartTimer();
void StopTimer();
long GetTime();

/*
namespace himan
{

namespace plugin
{
*/
__global__ void tpot_kernel(float* Tin, float* Pin, float* TPout, int N)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		//TPout[idx] = 273.15f + Tin[idx] * powf((1000 / Pin[idx]), 0.286f);
		
		// Assume K for now (since it actually is K when read from grib)
		
		if (Tin[idx] == kFloatMissing || Pin[idx] == kFloatMissing)
		{
		  TPout[idx] = kFloatMissing;
		}
		else
		{		
		  TPout[idx] = Tin[idx] * powf((1000 / Pin[idx]), 0.286f);
		}
	}
}


void tpot_cuda(const float* Tin, const float* Pin, float* TPout, int N)
{

	// Allocate host arrays and convert input data to float
	
	size_t size = N * sizeof(float);
	
	float* hP = (float*) malloc(size);

	size_t i = 0;
	
	for (i = 0; i < N; i++)
	{
		hP[i] = 850.f; // Hard coded pressure level 850
		
		TPout[i] = kFloatMissing;
	}
	
	// Allocate device arrays
	
	float* dT; 
	cudaMalloc((void **) &dT, size);
	checkCUDAError("malloc");
	
	float* dP;
  	cudaMalloc((void **) &dP, size);
  	checkCUDAError("malloc");

  	float *dTP;
  	cudaMalloc((void **) &dTP, size);
  	checkCUDAError("malloc");
  
    cudaMemcpy(dT, Tin, size, cudaMemcpyHostToDevice);
  	checkCUDAError("memcpy");
  
  	cudaMemcpy(dP, Pin, size, cudaMemcpyHostToDevice);
  	checkCUDAError("memcpy");
  
  	cudaMemcpy(dTP, TPout, size, cudaMemcpyHostToDevice);
  	checkCUDAError("memcpy");
  	
    // dims
     	
    const int n_threads_per_block = 512;
    int n_blocks = N/n_threads_per_block + (N%n_threads_per_block == 0?0:1);  	
  	
  	std::cout << "threads_per_block: " << n_threads_per_block << " number of blocks " << n_blocks << std::endl;
  	
  	dim3 dimGrid(n_blocks);
  	dim3 dimBlock(n_threads_per_block);
    
  	StartTimer();
  
  	tpot_kernel <<< dimGrid, dimBlock >>> (dT, dT, dTP, N);
  
  	// block until the device has completed
  	cudaThreadSynchronize();

  	// check if kernel execution generated an error

  	checkCUDAError("kernel invocation");
  
  	// Retrieve result from device 
  	cudaMemcpy(TPout, dTP, size, cudaMemcpyDeviceToHost);
  
  	checkCUDAError("memcpy");
  
  	StopTimer();
  
  	printf ("Calculation and data transfer took %ld microseconds on GPU\n", GetTime());
  	
  	cudaFree(dT);
  	cudaFree(dP);
  	cudaFree(dTP);
  	
}

void checkCUDAError(const std::string& msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        std::cout << "Cuda error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }                        
}

void StartTimer() {
  clock_gettime(CLOCK_REALTIME, &start_ts);
}

void StopTimer() {
  clock_gettime(CLOCK_REALTIME, &stop_ts);
}

long GetTime() {
  return ((stop_ts.tv_sec*1e9 + stop_ts.tv_nsec) - (start_ts.tv_sec*1e9 + start_ts.tv_nsec))/1e3;
}

/*
} // namespace plugin
} // namespace himan

*/