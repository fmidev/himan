// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "cuda_extern.h"
#include "cuda_helper.h"

// #include "cuPrintf.cu"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

const float kRadToDeg = 57.295775f; // 180 / PI
const float kDegToRad = 0.017453f; // PI / 180

namespace himan
{

namespace plugin
{

namespace windvector_cuda
{

__global__ void kernel_windvector(windvector_cuda_options opts, float* dU, float* dV, float* dataOut);
__global__ void kernel_windvector_rotation(windvector_cuda_options opts, float* dU, float* dV, float* dataOut);

__device__ void Calculate(float* __restrict__ dU, float* __restrict__ dV, float* __restrict__ dataOut, size_t N, bool vectorCalculation, bool dirCalculation);
__device__ void UVToEarthRelative(float* __restrict__ dU, float* __restrict__ dV, float firstLatitude, float firstLongitude, float di, float dj, float southPoleLat, float southPoleLon, size_t sizeY, size_t sizeX);

} // namespace windvector
} // namespace plugin
} // namespace himan

/*
 * Calculate results. At this point it as assumed that U and V are in correct form.
 *
 * Results will be placed in an array that's 1x or 3x the size of the grid in question.
 * Elements from 0..N will be speed, from N..2N is reserved for direction and from
 * 2N..3N are for windvector (if that's calculated).
 */

__device__ void himan::plugin::windvector_cuda::Calculate(float* __restrict__ dU, float* __restrict__ dV, float* __restrict__ dDataOut, size_t N, bool vectorCalculation, bool dirCalculation)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float U = dU[idx], V = dV[idx];
	
	if (U == kFloatMissing || V == kFloatMissing)
	{
		dDataOut[idx] = kFloatMissing;

		if (dirCalculation)
		{
			dDataOut[idx+N] = kFloatMissing;
		}

		if (vectorCalculation)
		{
			dDataOut[idx+2*N] = kFloatMissing;
		}
	}
	else
	{
		
		float speed = sqrtf(U*U + V*V);

		dDataOut[idx] = speed;

		float dir = 0.f;	// Direction is float although we round the result so that it *could* be int as well.
							// This is because if we use int the windvector calculation will have a small bias due
							// to int decimal value truncation.

		if (dirCalculation)
		{

			int offset = 180.f;

			if (!vectorCalculation)
			{
				// vector is calculated only for air, and for air we have offset of 180 degrees
				offset = 0.f;
			}

			if (speed > 0.f)
			{
				dir = kRadToDeg * atan2(U,V) + offset;

				// reduce the angle
				dir = fmod(dir, 360.f);

				// force it to be the positive remainder, so that 0 <= dir < 360
				dir = fmod((dir + 360.f), 360.f);

			}

			dDataOut[idx+N] = round(dir);
		}

		if (vectorCalculation)
		{
			dDataOut[idx+2*N] = round(dir/10.f) + 100.f * round(speed);
		}

	}
}

/*
 * Rotate U and V vectors that are grid-relative (U is pointing to grid north, V to grid east) to earth
 * relative form (U points to earth or map north etc). This requires that we first get the regular coordinates
 * of the rotated coordinates.
 *
 * NOTE! This function implicitly assumes that projection is rotated lat lon!
 *
 */

__device__ void himan::plugin::windvector_cuda::UVToEarthRelative(float* __restrict__ dU, float* __restrict__ dV,
									float firstLatitude, float firstLongitude, float di, float dj,
									float southPoleLat, float southPoleLon, size_t sizeY, size_t sizeX)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float U = dU[idx];
	float V = dV[idx];

	if (U != kFloatMissing && V != kFloatMissing)
	{
		int j = floorf(idx/sizeX);
		int i = idx - j * sizeX;

		float lon = firstLongitude + i * di;
		float lat = firstLatitude + j * dj;

		float SinYPole = sinf((southPoleLat + 90.f) * kDegToRad);
		float CosYPole = cosf((southPoleLat + 90.f) * kDegToRad);

		float SinXRot, CosXRot, SinYRot, CosYRot;

		sincosf(lon*kDegToRad, &SinXRot, &CosXRot);
		sincosf(lat*kDegToRad, &SinYRot, &CosYRot);

		float SinYReg = CosYPole * SinYRot + SinYPole * CosYRot * CosXRot;

		SinYReg = MIN(MAX(SinYReg, -1.f), 1.f);

		float YReg = asinf(SinYReg) * kRadToDeg;

		float CosYReg = cosf(YReg*kDegToRad);
		float CosXReg = (CosYPole * CosYRot * CosXRot - SinYPole * SinYRot) / CosYReg;

		CosXReg = MIN(MAX(CosXReg, -1.f), 1.f);
		float SinXReg = CosYRot * SinXRot / CosYReg;

		float XReg = acosf(CosXReg) * kRadToDeg;

		if (SinXReg < 0.f)
			XReg = -XReg;

		XReg += southPoleLon;

		// UV to earth relative

		float zxmxc = kDegToRad * (XReg - southPoleLon);

		float sinxmxc, cosxmxc;

		sincosf(zxmxc, &sinxmxc, &cosxmxc);

		float PA = cosxmxc * CosXRot + CosYPole * sinxmxc * SinXRot;
		float PB = CosYPole * sinxmxc * CosXRot * SinYRot + SinYPole * sinxmxc * CosYRot - cosxmxc * SinXRot * SinYRot;
		float PC = (-SinYPole) * SinXRot / CosYReg;
		float PD = (CosYPole * CosYRot - SinYPole * CosXRot * SinYRot) / CosYReg;

		float newU = PA * U + PB * V;
		float newV = PC * U + PD * V;

		dU[idx] = newU;
		dV[idx] = newV;
	}
}

__global__ void himan::plugin::windvector_cuda::kernel_windvector(windvector_cuda_options opts, float* dU, float* dV, float* dDataOut)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < opts.sizeX*opts.sizeY)
	{
		Calculate(dU, dV, dDataOut, opts.sizeX*opts.sizeY, opts.vectorCalculation, opts.dirCalculation);
	}
}

__global__ void himan::plugin::windvector_cuda::kernel_windvector_rotation(windvector_cuda_options opts, float* dU, float* dV, float* dDataOut)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.sizeX*opts.sizeY)
	{
		UVToEarthRelative(dU, dV, opts.firstLatitude, opts.firstLongitude, opts.di, opts.dj, opts.southPoleLat, opts.southPoleLon, opts.sizeY, opts.sizeX);
		Calculate(dU, dV, dDataOut, opts.sizeX*opts.sizeY, opts.vectorCalculation, opts.dirCalculation);
	}
}


void himan::plugin::windvector_cuda::DoCuda(windvector_cuda_options& opts)
{

	cudaSetDevice(opts.CudaDeviceIndex);
	CheckCudaError("deviceset");

	// Allocate host arrays and convert input data to float

	size_t N = opts.sizeY*opts.sizeX;

	size_t memSize = N * sizeof(float);

	// Allocate device arrays

	float* dU;
	cudaMalloc((void **) &dU, memSize);
	CheckCudaError("malloc dU");

	float* dV;

	cudaMalloc((void **) &dV, memSize);
	CheckCudaError("malloc dV");

	float *dDataOut;

	int numberOfParams = 2;

	if (opts.vectorCalculation)
	{
		numberOfParams = 3;
	}
	else if (!opts.dirCalculation)
	{
		numberOfParams = 1;
	}

	cudaMalloc((void **) &dDataOut, numberOfParams*memSize);

	CheckCudaError("malloc dDataOut");

	cudaMemcpy(dU, opts.Uin, memSize, cudaMemcpyHostToDevice);
	CheckCudaError("memcpy Uin");

	cudaMemcpy(dV, opts.Vin, memSize, cudaMemcpyHostToDevice);
	CheckCudaError("memcpy Vin");

	// dims

	const int blockSize = 512;
	const int gridSize = N/blockSize + (N%blockSize == 0?0:1);

	dim3 gridDim(gridSize);
	dim3 blockDim(blockSize);

	// cudaPrintfInit();

	// Better do this once here than millions of times in the kernel

	if (opts.southPoleLat > 0)
	{
		opts.southPoleLat = -opts.southPoleLat;
		opts.southPoleLon = 0;
	}
	if (opts.needRotLatLonGridRotation || opts.dirCalculation)
	{	
		kernel_windvector_rotation <<< gridDim, blockDim >>> (opts, dU, dV, dDataOut);
	}
	else
	{
		kernel_windvector <<< gridDim, blockDim >>> (opts, dU, dV, dDataOut);
	}

	// block until the device has completed
	cudaDeviceSynchronize();

	// check if kernel execution generated an error

	CheckCudaError("kernel invocation");

	// cudaPrintfDisplay(stdout, true);
	// cudaPrintfEnd();

	// Retrieve result from device
	cudaMemcpy(opts.dataOut, dDataOut, numberOfParams*memSize, cudaMemcpyDeviceToHost);

	CheckCudaError("memcpy");

	cudaFree(dU);
	cudaFree(dV);
	cudaFree(dDataOut);
	
}
