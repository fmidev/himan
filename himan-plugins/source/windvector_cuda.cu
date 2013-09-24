// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "windvector_cuda.h"
#include "cuda_helper.h"
// #include "cuPrintf.cu"

#define BitMask1(i)	(1u << i)
#define BitTest(n,i)	!!((n) & BitMask1(i))

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

__global__ void Rotate(double* __restrict__ dU, double* __restrict__ dV, windvector_cuda_options opts);
__global__ void Calculate(const double* __restrict__ dU,
							const double* __restrict__ dV,
							double* __restrict__ dSpeed,
							double* __restrict__ dDir,
							double* __restrict__ dVector,
							windvector_cuda_options opts, int* dMissingValueCount);

} // namespace windvector_cuda
} // namespace plugin
} // namespace himan

/*
 * Calculate results. At this point it as assumed that U and V are in correct form.
 */

__global__ void himan::plugin::windvector_cuda::Calculate(const double* __restrict__ dU,
															const double* __restrict__ dV,
															double* __restrict__ dSpeed,
															double* __restrict__ dDir,
															double* __restrict__ dVector,
															windvector_cuda_options opts,
															int* dMissingValuesCount)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.sizeX*opts.sizeY)
	{
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

			atomicAdd(dMissingValuesCount, 1);

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

					// modulo operator is supposedly slow on cuda ?

					/*
					 * quote:
					 * 
					 * Integer division and modulo operation are costly: tens of instructions on devices of
					 * compute capability 1.x, below 20 instructions on devices of compute capability 2.x and
					 * higher.
					 */

					// reduce the angle
					while (dir > 360)
					{
						dir -= 360;
					}
					
					// force it to be the positive remainder, so that 0 <= dir < 360

					while (dir < 0)
					{
						dir += 360;
					}


				}

				dDir[idx] = round(dir);
			}

			if (opts.vectorCalculation)
			{
				dVector[idx] = round(dir/10) + 100 * round(speed);
			}
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

__global__ void himan::plugin::windvector_cuda::Rotate(double* __restrict__ dU, double* __restrict__ dV, windvector_cuda_options opts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.sizeX*opts.sizeY)
	{

		double U = dU[idx];
		double V = dV[idx];

		if (U != himan::kFloatMissing && V != himan::kFloatMissing)
		{
			int j;
			
			if (opts.jScansPositive)
			{
				j = floor(static_cast<double> (idx/opts.sizeX));
			}
			else
			{
				j = opts.sizeY - floor(static_cast<double> (idx/opts.sizeX));
			}

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
}

void himan::plugin::windvector_cuda::DoCuda(windvector_cuda_options& opts, windvector_cuda_data& datas)
{

	// Allocate host arrays and convert input data to double

	size_t N = opts.sizeY*opts.sizeX;

	// Allocate device arrays

	double* dU = 0;
	double* dV = 0;
	double* dSpeed = 0;
	double* dDir = 0;
	double* dVector = 0;
	
	int* dMissingValuesCount = 0;

	CUDA_CHECK(cudaMalloc((void **) &dMissingValuesCount, sizeof(int)));

	CUDA_CHECK(cudaHostGetDevicePointer(&dSpeed, datas.speed, 0));

	if (opts.targetType != kGust)
	{
		CUDA_CHECK(cudaHostGetDevicePointer(&dDir, datas.dir, 0));
	}

	if (opts.vectorCalculation)
	{
		CUDA_CHECK(cudaHostGetDevicePointer(&dVector, datas.vector, 0));
	}

	size_t memsize = opts.sizeX*opts.sizeY*sizeof(double);

	if (opts.pU)
	{
		CUDA_CHECK(cudaHostGetDevicePointer(&dU, datas.u, 0));
	}
	else
	{
		CUDA_CHECK(cudaMalloc((void **) &dU, memsize));
		CUDA_CHECK(cudaMemcpy(dU, datas.u, memsize, cudaMemcpyHostToDevice));
	}

	if (opts.pV)
	{
		CUDA_CHECK(cudaHostGetDevicePointer(&dV, datas.v, 0));
	}
	else
	{
		CUDA_CHECK(cudaMalloc((void **) &dV, memsize));
		CUDA_CHECK(cudaMemcpy(dV, datas.v, memsize, cudaMemcpyHostToDevice));
	}
	
	int src=0;

	CUDA_CHECK(cudaMemcpy(dMissingValuesCount, &src, sizeof(int), cudaMemcpyHostToDevice));

	// dims

	const int blockSize = 128;
	const int gridSize = N/blockSize + (N%blockSize == 0?0:1);

	// Better do this once here than millions of times in the kernel

	if (opts.southPoleLat > 0)
	{
		opts.southPoleLat = -opts.southPoleLat;
		opts.southPoleLon = 0;
	}

	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	if (opts.pU)
	{
		datas.pU->Unpack(dU, &stream);
	}

	if (opts.pV)
	{
		datas.pV->Unpack(dV, &stream);
	}

	/*
	 *  If calculating gust, do not ever rotate grid since we don't calculate
	 * direction for it.
	*/

	if (opts.targetType != kGust && opts.needRotLatLonGridRotation)
	{
		Rotate <<< gridSize, blockSize, 0, stream >>> (dU, dV, opts);
	}
	
	Calculate <<< gridSize, blockSize, 0, stream >>> (dU, dV, dSpeed, dDir, dVector, opts, dMissingValuesCount);

	// block until the stream has completed
	CUDA_CHECK(cudaStreamSynchronize(stream));

	// check if kernel execution generated an error

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	CUDA_CHECK(cudaMemcpy(&opts.missingValuesCount, dMissingValuesCount, sizeof(int), cudaMemcpyDeviceToHost));

	if (!opts.pU)
	{
		CUDA_CHECK(cudaFree(dU));
	}

	if (!opts.pV)
	{
		CUDA_CHECK(cudaFree(dV));
	}

	CUDA_CHECK(cudaFree(dMissingValuesCount));

	CUDA_CHECK(cudaStreamDestroy(stream));
}
