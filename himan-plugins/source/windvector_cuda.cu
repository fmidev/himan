// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "windvector_cuda.h"
#include "cuda_helper.h"

 //#include "cuPrintf.cu"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

const double kRadToDeg = 57.29577951307855; // 180 / PI
const double kDegToRad = 0.017453292519944; // PI / 180

namespace himan
{

namespace plugin
{

namespace windvector_cuda
{

__global__ void UnpackAndCalculate(const unsigned char* dUPacked,
									const unsigned char* dVPacked,
									double* dU,
									double* dV,
									double* dSpeed,
									double* dDir,
									double* dVector,
									windvector_cuda_options opts,
									int* dMissingValuesCount);

__global__ void Calculate(double* dU, double* dV, double* dSpeed, double* dDir, double* dVector, windvector_cuda_options opts, int* dMissingValueCount);

__device__ void Rotate(double* __restrict__ dU, double* __restrict__ dV, windvector_cuda_options opts, int idx);
__device__ void _Calculate(double* __restrict__ dU, 
							double* __restrict__ dV,
							double* __restrict__ dSpeed,
							double* __restrict__ dDir,
							double* __restrict__ dVector,
							windvector_cuda_options opts, int* dMissingValueCount, int idx);

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

__device__ void himan::plugin::windvector_cuda::_Calculate(double* __restrict__ dU,
															double* __restrict__ dV,
															double* __restrict__ dSpeed,
															double* __restrict__ dDir,
															double* __restrict__ dVector,
															windvector_cuda_options opts, int* dMissingValuesCount, int idx)
{

	size_t N = opts.sizeX * opts.sizeY;
	
	double U = dU[idx], V = dV[idx];

	if (U == kFloatMissing || V == kFloatMissing)
	{
		dSpeed[idx] = kFloatMissing;

		if (opts.targetType != kGust)
		{
			dDir[idx] = kFloatMissing;
		}

		if (opts.vectorCalculation)
		{
			dVector[idx] = kFloatMissing;
		}
	}
	else
	{
		
		double speed = sqrt(U*U + V*V);

		dSpeed[idx] = speed;

		double dir = 0;	// Direction is double although we round the result so that it *could* be int as well.
							// This is because if we use int the windvector calculation will have a small bias due
							// to int decimal value truncation.

		if (opts.targetType != kGust)
		{

			int offset = 180;

			if (opts.targetType == kSea || opts.targetType == kIce)
			{
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

			dDir[idx] = round(dir);
		}

		if (opts.vectorCalculation)
		{
			dVector[idx+2*N] = round(dir/10) + 100 * round(speed);
		}

	}
}

__global__ void himan::plugin::windvector_cuda::UnpackAndCalculate(const unsigned char* dUPacked,
									const unsigned char* dVPacked,
									double* dU,
									double* dV,
									double* dSpeed,
									double* dDir,
									double* dVector,
									windvector_cuda_options opts,
									int* dMissingValuesCount)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.sizeX*opts.sizeY)
	{
		if (opts.simplePackedU.HasData())
		{
			SimpleUnpack(dUPacked, dU, opts.sizeX*opts.sizeY, opts.simplePackedU.bitsPerValue, opts.simplePackedU.binaryScaleFactor, opts.simplePackedU.decimalScaleFactor, opts.simplePackedU.referenceValue, idx);
		}

		if (opts.simplePackedV.HasData())
		{
			SimpleUnpack(dVPacked, dV, opts.sizeX*opts.sizeY, opts.simplePackedV.bitsPerValue, opts.simplePackedV.binaryScaleFactor, opts.simplePackedV.decimalScaleFactor, opts.simplePackedV.referenceValue, idx);
		}

		/*
		 *  If calculating gust, do not ever rotate grid since we don't calculate
		 * direction for it.
		 */

		if (opts.targetType != kGust && opts.needRotLatLonGridRotation)
		{
			Rotate(dU, dV, opts, idx);
		}

		_Calculate(dU, dV, dSpeed, dDir, dVector, opts, dMissingValuesCount, idx);
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

__device__ void himan::plugin::windvector_cuda::Rotate(double* __restrict__ dU, double* __restrict__ dV, windvector_cuda_options opts, int idx)
{
	double U = dU[idx];
	double V = dV[idx];

	if (U != kFloatMissing && V != kFloatMissing)
	{
		int j = floor(static_cast<double> (idx/opts.sizeX));
		int i = idx - j * opts.sizeX;

		double lon = opts.firstLongitude + i * opts.di;
		double lat = opts.firstLatitude + j * opts.dj;


		double SinYPole = sin((opts.southPoleLat + 90.) * kDegToRad);
		double CosYPole = cos((opts.southPoleLat + 90.) * kDegToRad);

		double SinXRot, CosXRot, SinYRot, CosYRot;

		sincos(lon*kDegToRad, &SinXRot, &CosXRot);
		sincos(lat*kDegToRad, &SinYRot, &CosYRot);

		double SinYReg = CosYPole * SinYRot + SinYPole * CosYRot * CosXRot;

		SinYReg = MIN(MAX(SinYReg, -1.), 1.);

		double YReg = asin(SinYReg) * kRadToDeg;

		double CosYReg = cos(YReg*kDegToRad);

		double CosXReg = (CosYPole * CosYRot * CosXRot - SinYPole * SinYRot) / CosYReg;

		CosXReg = MIN(MAX(CosXReg, -1.), 1.);
		double SinXReg = CosYRot * SinXRot / CosYReg;

		double XReg = acos(CosXReg) * kRadToDeg;

		if (SinXReg < 0.)
			XReg = -XReg;

		XReg += opts.southPoleLon;

		// UV to earth relative

		double zxmxc = kDegToRad * (XReg - opts.southPoleLon);

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

__global__ void himan::plugin::windvector_cuda::Calculate(double* dU, double* dV, double* dSpeed, double* dDir, double* dVector, windvector_cuda_options opts, int* dMissingValueCount)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < opts.sizeX*opts.sizeY)
	{

		/*
		 *  If calculating gust, do not ever rotate grid since we don't calculate
		 * direction for it.
		 */

		if (opts.targetType != kGust && opts.needRotLatLonGridRotation)
		{
			Rotate(dU, dV, opts, idx);
		}

		_Calculate(dU, dV, dSpeed, dDir, dVector, opts, dMissingValueCount, idx);
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
	double* dSpeed;
	double* dDir;
	double* dVector;

	unsigned char* dUPacked;
	unsigned char* dVPacked;
	
	int* dMissingValuesCount;

	CUDA_CHECK(cudaMalloc((void **) &dMissingValuesCount, sizeof(int)));

	CUDA_CHECK(cudaMalloc((void **) &dU, memSize));
	CUDA_CHECK(cudaMalloc((void **) &dV, memSize));

	CUDA_CHECK(cudaMalloc((void **) &dSpeed, memSize));

	if (opts.targetType != kGust)
	{
		CUDA_CHECK(cudaMalloc((void **) &dDir, memSize));
	}

	if (opts.vectorCalculation)
	{
		CUDA_CHECK(cudaMalloc((void **) &dVector, memSize));
	}
	
	if (opts.simplePackedU.HasData())
	{
		CUDA_CHECK(cudaMalloc((void **) &dUPacked, opts.simplePackedU.dataLength * sizeof(unsigned char)));
		CUDA_CHECK(cudaMemcpy(dUPacked, opts.simplePackedU.data, opts.simplePackedU.dataLength * sizeof(unsigned char), cudaMemcpyHostToDevice));
	}
	else
	{
		CUDA_CHECK(cudaMemcpy(dU, opts.UIn, memSize, cudaMemcpyHostToDevice));
	}

	if (opts.simplePackedV.HasData())
	{
		CUDA_CHECK(cudaMalloc((void **) &dVPacked, opts.simplePackedV.dataLength * sizeof(unsigned char)));
		CUDA_CHECK(cudaMemcpy(dVPacked, opts.simplePackedV.data, opts.simplePackedV.dataLength * sizeof(unsigned char), cudaMemcpyHostToDevice));
	}
	else
	{
		CUDA_CHECK(cudaMemcpy(dV, opts.VIn, memSize, cudaMemcpyHostToDevice));
	}

	int src=0;

	CUDA_CHECK(cudaMemcpy(dMissingValuesCount, &src, sizeof(int), cudaMemcpyHostToDevice));

	// dims

	const int blockSize = 512;
	const int gridSize = N/blockSize + (N%blockSize == 0?0:1);

	dim3 gridDim(gridSize);
	dim3 blockDim(blockSize);

	 //cudaPrintfInit();

	// Better do this once here than millions of times in the kernel

	if (opts.southPoleLat > 0)
	{
		opts.southPoleLat = -opts.southPoleLat;
		opts.southPoleLon = 0;
	}

	if (opts.isPackedData)
	{
		UnpackAndCalculate <<< gridDim, blockDim >>> (dUPacked, dVPacked, dU, dV, dSpeed, dDir, dVector, opts, dMissingValuesCount);
	}
	else
	{
		Calculate <<< gridDim, blockDim >>> (dU, dV, dSpeed, dDir, dVector, opts, dMissingValuesCount);
	}

	// block until the device has completed
	CUDA_CHECK(cudaDeviceSynchronize());

	// check if kernel execution generated an error

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	 //cudaPrintfDisplay(stdout, true);
	 //cudaPrintfEnd();

	// Retrieve result from device

	CUDA_CHECK(cudaMemcpy(opts.speed, dSpeed, memSize, cudaMemcpyDeviceToHost));
	
	if (opts.targetType != kGust)
	{
		CUDA_CHECK(cudaMemcpy(opts.dir, dDir, memSize, cudaMemcpyDeviceToHost));
	}

	if (opts.vectorCalculation)
	{
		CUDA_CHECK(cudaMemcpy(opts.vector, dVector, memSize, cudaMemcpyDeviceToHost));
	}

	CUDA_CHECK(cudaFree(dU));
	CUDA_CHECK(cudaFree(dV));
	CUDA_CHECK(cudaFree(dMissingValuesCount));

	if (opts.simplePackedU.HasData())
	{
		CUDA_CHECK(cudaFree(dUPacked));
	}

	if (opts.simplePackedV.HasData())
	{
		CUDA_CHECK(cudaFree(dVPacked));
	}

	
}
