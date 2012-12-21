// System includes
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

void tpot_cuda(double* Tin, double* Pin, double* TPout, int N)
{
	printf("I'm actually at tpot_cuda.cu!\n");	

}

__global__ void tpot_kernel(float* Tin, float* Pin, float* TPout, int N)
{

  int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
  if (idx<N)
  {
	TPout[idx] = 273.15+Tin[idx] * powf((1000/Pin[idx]),0.286);
  }

}