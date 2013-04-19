// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "windvector_cuda.h"
#include "cuda_helper.h"

// #include "cuPrintf.cu"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

const double kRadToDeg = 57.295775f; // 180 / PI
const double kDegToRad = 0.017453f; // PI / 180

namespace himan
{

namespace plugin
{

namespace windvector_cuda
{

__global__ void kernel_windvector(windvector_cuda_options opts, double* dU, double* dV, double* dataOut);
__global__ void kernel_windvector_rotation(windvector_cuda_options opts, double* dU, double* dV, double* dataOut);

__device__ void Calculate(double* __restrict__ dU, double* __restrict__ dV, double* __restrict__ dataOut, size_t N, bool vectorCalculation, bool dirCalculation);
__device__ void UVToEarthRelative(double* __restrict__ dU, double* __restrict__ dV, double firstLatitude, double firstLongitude, double di, double dj, double southPoleLat, double southPoleLon, size_t sizeY, size_t sizeX);

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

__device__ void himan::plugin::windvector_cuda::Calculate(double* __restrict__ dU, double* __restrict__ dV, double* __restrict__ dDataOut, size_t N, bool vectorCalculation, bool dirCalculation)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	double U = dU[idx], V = dV[idx];
	
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
		
		double speed = sqrtf(U*U + V*V);

		dDataOut[idx] = speed;

		double dir = 0;	// Direction is double although we round the result so that it *could* be int as well.
							// This is because if we use int the windvector calculation will have a small bias due
							// to int decimal value truncation.

		if (dirCalculation)
		{

			int offset = 180;

			if (!vectorCalculation)
			{
				// vector is calculated only for air, and for air we have offset of 180 degrees
				offset = 0;
			}

			if (speed > 0)
			{
				dir = kRadToDeg * atan2(U,V) + offset;

				// reduce the angle
				dir = fmod(dir, 360);

				// force it to be the positive remainder, so that 0 <= dir < 360
				dir = fmod((dir + 360), 360);

			}

			dDataOut[idx+N] = round(dir);
		}

		if (vectorCalculation)
		{
			dDataOut[idx+2*N] = round(dir/10) + 100 * round(speed);
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

__device__ void himan::plugin::windvector_cuda::UVToEarthRelative(double* __restrict__ dU, double* __restrict__ dV,
									double firstLatitude, double firstLongitude, double di, double dj,
									double southPoleLat, double southPoleLon, size_t sizeY, size_t sizeX)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	double U = dU[idx];
	double V = dV[idx];

	if (U != kFloatMissing && V != kFloatMissing)
	{
		int j = floor(idx/sizeX);
		int i = idx - j * sizeX;

		double lon = firstLongitude + i * di;
		double lat = firstLatitude + j * dj;

		double SinYPole = sin((southPoleLat + 90) * kDegToRad);
		double CosYPole = cos((southPoleLat + 90) * kDegToRad);

		double SinXRot, CosXRot, SinYRot, CosYRot;

		sincos(lon*kDegToRad, &SinXRot, &CosXRot);
		sincos(lat*kDegToRad, &SinYRot, &CosYRot);

		double SinYReg = CosYPole * SinYRot + SinYPole * CosYRot * CosXRot;

		SinYReg = MIN(MAX(SinYReg, -1), 1);

		double YReg = asin(SinYReg) * kRadToDeg;

		double CosYReg = cos(YReg*kDegToRad);
		double CosXReg = (CosYPole * CosYRot * CosXRot - SinYPole * SinYRot) / CosYReg;

		CosXReg = MIN(MAX(CosXReg, -1), 1);
		double SinXReg = CosYRot * SinXRot / CosYReg;

		double XReg = acos(CosXReg) * kRadToDeg;

		if (SinXReg < 0)
			XReg = -XReg;

		XReg += southPoleLon;

		// UV to earth relative

		double zxmxc = kDegToRad * (XReg - southPoleLon);

		double sinxmxc, cosxmxc;

		sincos(zxmxc, &sinxmxc, &cosxmxc);

		double PA = cosxmxc * CosXRot + CosYPole * sinxmxc * SinXRot;
		double PB = CosYPole * sinxmxc * CosXRot * SinYRot + SinYPole * sinxmxc * CosYRot - cosxmxc * SinXRot * SinYRot;
		double PC = (-SinYPole) * SinXRot / CosYReg;
		double PD = (CosYPole * CosYRot - SinYPole * CosXRot * SinYRot) / CosYReg;

		double newU = PA * U + PB * V;
		double newV = PC * U + PD * V;

		dU[idx] = newU;
		dV[idx] = newV;
	}
}

__global__ void himan::plugin::windvector_cuda::kernel_windvector(windvector_cuda_options opts, double* dU, double* dV, double* dDataOut)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < opts.sizeX*opts.sizeY)
	{
		Calculate(dU, dV, dDataOut, opts.sizeX*opts.sizeY, opts.vectorCalculation, opts.dirCalculation);
	}
}

__global__ void himan::plugin::windvector_cuda::kernel_windvector_rotation(windvector_cuda_options opts, double* dU, double* dV, double* dDataOut)
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

	CUDA_CHECK(cudaSetDevice(opts.cudaDeviceIndex));
	
	// Allocate host arrays and convert input data to double

	size_t N = opts.sizeY*opts.sizeX;

	size_t memSize = N * sizeof(double);

	// Allocate device arrays

	double* dU;
	double* dV;
	double* dDataOut;

	CUDA_CHECK(cudaMalloc((void **) &dU, memSize));
	CUDA_CHECK(cudaMalloc((void **) &dV, memSize));

	int numberOfParams = 2;

	if (opts.vectorCalculation)
	{
		numberOfParams = 3;
	}
	else if (!opts.dirCalculation)
	{
		numberOfParams = 1;
	}

	CUDA_CHECK(cudaMalloc((void **) &dDataOut, numberOfParams*memSize));
	
	CUDA_CHECK(cudaMemcpy(dU, opts.Uin, memSize, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dV, opts.Vin, memSize, cudaMemcpyHostToDevice));
	
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
	CUDA_CHECK(cudaDeviceSynchronize());

	// check if kernel execution generated an error

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	// cudaPrintfDisplay(stdout, true);
	// cudaPrintfEnd();

	// Retrieve result from device
	CUDA_CHECK(cudaMemcpy(opts.dataOut, dDataOut, numberOfParams*memSize, cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(dU));
	CUDA_CHECK(cudaFree(dV));
	CUDA_CHECK(cudaFree(dDataOut));
	
}
