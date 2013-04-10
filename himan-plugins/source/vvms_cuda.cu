// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "vvms_cuda.h"
#include "cuda_helper.h"

namespace himan
{

namespace plugin
{

namespace vvms_cuda
{

__global__ void UnpackAndCalculate();
__global__ void Calculate();

__global__ void kernel_constant_pressure(const double* __restrict__ dT, double TBase, double P, const double* __restrict__ dVVPas, double* __restrict__ VVout, size_t N);
__global__ void kernel_varying_pressure(const double* __restrict__ dT, double TBase, const double* __restrict__ dP, double PScale, const double* __restrict__ dVVPas, double* __restrict__ VVout, size_t N);


} // namespace tpot
} // namespace plugin
} // namespace himan

__global__ void himan::plugin::vvms_cuda::kernel_constant_pressure(const double* __restrict__ dT, double TBase, double P, const double* __restrict__ dVVPas, double* __restrict__ VVout, size_t N)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{

		if (dT[idx] == kFloatMissing || P == kFloatMissing || dVVPas[idx] == kFloatMissing)
		{
			VVout[idx] = kFloatMissing;
		}
		else
		{
			VVout[idx] = 287.f * -dVVPas[idx] * (TBase + dT[idx]) / (9.81f * P);
		}
	}
}

__global__ void himan::plugin::vvms_cuda::kernel_varying_pressure(const double* dT, double TBase, const double* __restrict__ dP, double PScale, const double* __restrict__ dVVPas, double* __restrict__ VVout, size_t N)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{

		if (dT[idx] == kFloatMissing || dP[idx] == kFloatMissing || dVVPas[idx] == kFloatMissing)
		{
			VVout[idx] = kFloatMissing;
		}
		else
		{
			VVout[idx] = 287.f * -dVVPas[idx] * (TBase + dT[idx]) / (9.81f * dP[idx] * PScale);
		}
	}
}

void himan::plugin::vvms_cuda::DoCuda(vvms_cuda_options& opts)
{

	CUDA_CHECK(cudaSetDevice(opts.cudaDeviceIndex));

	size_t memSize = opts.N * sizeof(double);

	// Allocate device arrays

	double* dT;
	double* dP;
	double *dVVPas;
	double *dVVOut;
	
	CUDA_CHECK(cudaMalloc((void **) &dT, memSize));

	if (!opts.isConstantPressure)
	{
		CUDA_CHECK(cudaMalloc((void **) &dP, memSize));
	}

	CUDA_CHECK(cudaMalloc((void **) &dVVPas, memSize));
	CUDA_CHECK(cudaMalloc((void **) &dVVOut, memSize));

	CUDA_CHECK(cudaMemcpy(dT, opts.TIn, memSize, cudaMemcpyHostToDevice));

	if (!opts.isConstantPressure)
	{
		CUDA_CHECK(cudaMemcpy(dP, opts.PIn, memSize, cudaMemcpyHostToDevice));
	}

	CUDA_CHECK(cudaMemcpy(dVVPas, opts.VVIn, memSize, cudaMemcpyHostToDevice));

	// dims

	const int blockSize = 512;
	const int gridSize = opts.N/blockSize + (opts.N%blockSize == 0?0:1);

	dim3 gridDim(gridSize);
	dim3 blockDim(blockSize);

	if (opts.isConstantPressure)
	{
		kernel_constant_pressure <<< gridDim, blockDim >>> (dT, opts.TBase, opts.PConst, dVVPas, dVVOut, opts.N);
	}
	else
	{
		kernel_varying_pressure <<< gridDim, blockDim >>> (dT, opts.TBase, dP, opts.PScale, dVVPas, dVVOut, opts.N);
	}

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	// block until the device has completed
	CUDA_CHECK(cudaDeviceSynchronize());

	// Retrieve result from device
	CUDA_CHECK(cudaMemcpy(opts.VVOut, dVVOut, memSize, cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(dT));

	if (!opts.isConstantPressure)
	{
		CUDA_CHECK(cudaFree(dP));
	}

	CUDA_CHECK(cudaFree(dVVPas));
	CUDA_CHECK(cudaFree(dVVOut));

}
