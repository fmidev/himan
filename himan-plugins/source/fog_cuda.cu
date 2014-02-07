// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include <cuda_helper.h>
#include <fog_cuda.h>

namespace himan
{

namespace plugin
{

namespace fog_cuda
{

	__global__ void Calculate(	const double* __restrict__ dDTC2M, 
								const double* __restrict__ dTKGround, 
								const double* __restrict__ dFF10M, 
								double* __restrict__ dF, 
								fog_cuda_options opts, 
								int* dMissingValuesCount);

} // namespace fog_cuda
} // namespace plugin
} // namespace himan

__global__ void himan::plugin::fog_cuda::Calculate( const double* __restrict__ dDTC2M, 
													const double* __restrict__ dTKGround, 
													const double* __restrict__ dFF10M, 
													double* __restrict__ dF, 
													fog_cuda_options opts, 
													int* dMissingValuesCount )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		if (dDTC2M[idx] == kFloatMissing || dTKGround[idx] == kFloatMissing || dFF10M[idx] == kFloatMissing )
		{
			atomicAdd(dMissingValuesCount, 1);
			dF[idx] = kFloatMissing;
		}
		else
		{
			dF[idx] = (dDTC2M[idx] - dTKGround[idx] > -0.3 && dFF10M[idx] < 5) ? 607 : 0;
			
		}
	}
}

void himan::plugin::fog_cuda::DoCuda(fog_cuda_options& opts, fog_cuda_data& datas)
{
#if 0
	CUDA_CHECK(cudaSetDevice(opts.cudaDeviceIndex));

	size_t memsize = opts.N * sizeof(double);

	// Allocate device arrays

	double* dDTC2M;
	double* dTKGround;
	double* dFF10M;
	
	double* dF;
	
	int* dMissingValuesCount;

	CUDA_CHECK(cudaMalloc((void **) &dMissingValuesCount, sizeof(int)));

	CUDA_CHECK(cudaHostGetDevicePointer(&dF, datas.F, 0));

	if (opts.pDTC2M)
	{
		CUDA_CHECK(cudaHostGetDevicePointer(&dDTC2M, datas.DTC2M, 0));
	}
	else
	{
		CUDA_CHECK(cudaMalloc((void **) &dDTC2M, memsize));
		CUDA_CHECK(cudaMemcpy(dDTC2M, datas.DTC2M, memsize, cudaMemcpyHostToDevice));
	}

	if (opts.pTKGround)
	{
		CUDA_CHECK(cudaHostGetDevicePointer(&dTKGround, datas.TKGround, 0));
	}
	else
	{
		CUDA_CHECK(cudaMalloc((void **) &dTKGround, memsize));
		CUDA_CHECK(cudaMemcpy(dTKGround, datas.TKGround, memsize, cudaMemcpyHostToDevice));
	}

	if (opts.pFF10M)
	{
		CUDA_CHECK(cudaHostGetDevicePointer(&dFF10M, datas.FF10M, 0));
	}
	else
	{
		CUDA_CHECK(cudaMalloc((void **) &dFF10M, memsize));
		CUDA_CHECK(cudaMemcpy(dFF10M, datas.FF10M, memsize, cudaMemcpyHostToDevice));
	}

	int src = 0;

	CUDA_CHECK(cudaMemcpy(dMissingValuesCount, &src, sizeof(int), cudaMemcpyHostToDevice));
	
	// dims

	const int blockSize = 512;
	const int gridSize = opts.N/blockSize + (opts.N%blockSize == 0?0:1);

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));
	
	if (opts.pDTC2M)
	{
		datas.pDTC2M->Unpack(dDTC2M, &stream);
	}

	if (opts.pTKGround)
	{
		datas.pTKGround->Unpack(dTKGround, &stream);
	}

	if (opts.pFF10M)
	{
		datas.pFF10M->Unpack(dFF10M, &stream);
	}

	Calculate <<< gridSize, blockSize, 0, stream >>> (dDTC2M, dTKGround, dFF10M, dF, opts, dMissingValuesCount);

	// block until the device has completed
	CUDA_CHECK(cudaStreamSynchronize(stream));

	// check if kernel execution generated an error

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	// Retrieve result from device
	CUDA_CHECK(cudaMemcpy(&opts.missingValuesCount, dMissingValuesCount, sizeof(int), cudaMemcpyDeviceToHost));

	if (!opts.pDTC2M)
	{
		CUDA_CHECK(cudaFree(dDTC2M));
	}

	if (!opts.pTKGround)
	{
		CUDA_CHECK(cudaFree(dTKGround));
	}

	if (!opts.pFF10M)
	{
		CUDA_CHECK(cudaFree(dFF10M));
	}

	CUDA_CHECK(cudaFree(dMissingValuesCount));
	CUDA_CHECK(cudaStreamDestroy(stream));
#endif
}
