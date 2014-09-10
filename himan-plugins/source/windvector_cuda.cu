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

/*
 * Calculate results. At this point it as assumed that U and V are in correct form.
 */

__global__ void himan::plugin::windvector_cuda::Calculate(const double* __restrict__ d_u,
															const double* __restrict__ d_v,
															double* __restrict__ d_speed,
															double* __restrict__ d_dir,
															double* __restrict__ d_vector,
															options opts,
															int* d_missing)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		double U = d_u[idx], V = d_v[idx];

		if (U == kFloatMissing || V == kFloatMissing)
		{
			d_speed[idx] = kFloatMissing;

			if (opts.target_type != kGust)
			{
				d_dir[idx] = kFloatMissing;
			}

			if (opts.vector_calculation)
			{
				d_vector[idx] = kFloatMissing;
			}

			atomicAdd(d_missing, 1);

		}
		else
		{

			double speed = sqrt(U*U + V*V);

			d_speed[idx] = speed;

			double dir = 0;	// Direction is double although we round the result so that it *could* be int as well.
								// This is because if we use int the windvector calculation will have a small bias due
								// to int decimal value truncation.

			if (opts.target_type != kGust)
			{

				int offset = 180;

				if (opts.target_type == kSea || opts.target_type == kIce)
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

				d_dir[idx] = round(dir);
			}

			if (opts.vector_calculation)
			{
				d_vector[idx] = round(dir/10) + 100 * round(speed);
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

__global__ void himan::plugin::windvector_cuda::Rotate(double* __restrict__ d_u, double* __restrict__ d_v, info_simple opts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.size_x * opts.size_y)
	{

		double U = d_u[idx];
		double V = d_v[idx];

		if (U != himan::kFloatMissing && V != himan::kFloatMissing)
		{
			int j;
			
			if (opts.j_scans_positive)
			{
				j = floor(static_cast<double> (idx/opts.size_x));
			}
			else
			{
				j = opts.size_y - floor(static_cast<double> (idx/opts.size_x));
			}

			int i = idx - j * opts.size_x;

			double lon = opts.first_lon + i * opts.di;
			double lat = opts.first_lat + j * opts.dj;

			double SinYPole = sin((opts.south_pole_lat + 90.) * kDegToRad);
			double CosYPole = cos((opts.south_pole_lat + 90.) * kDegToRad);

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
			{
				XReg = -XReg;
			}

			XReg += opts.south_pole_lon;

			// UV to earth relative

			double zxmxc = kDegToRad * (XReg - opts.south_pole_lon);

			double sinxmxc, cosxmxc;

			sincos(zxmxc, &sinxmxc, &cosxmxc);

			double PA = cosxmxc * CosXRot + CosYPole * sinxmxc * SinXRot;
			double PB = CosYPole * sinxmxc * CosXRot * SinYRot + SinYPole * sinxmxc * CosYRot - cosxmxc * SinXRot * SinYRot;
			double PC = (-SinYPole) * SinXRot / CosYReg;
			double PD = (CosYPole * CosYRot - SinYPole * CosXRot * SinYRot) / CosYReg;

			double newU = PA * U + PB * V;
			double newV = PC * U + PD * V;

			d_u[idx] = newU;
			d_v[idx] = newV;
		}
	}
}

void himan::plugin::windvector_cuda::Process(options& opts)
{

	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	// Allocate device arrays

	double* d_u = 0;
	double* d_v = 0;
	double* d_speed = 0;
	double* d_dir = 0;
	double* d_vector = 0;
	
	int* d_missing = 0;

	// Allocate memory on device

	size_t memsize = opts.N*sizeof(double);

	CUDA_CHECK(cudaMalloc((void **) &d_u, sizeof(double) * memsize));
	CUDA_CHECK(cudaMalloc((void **) &d_v, sizeof(double) * memsize));

	CUDA_CHECK(cudaMalloc((void **) &d_missing, sizeof(int)));
	CUDA_CHECK(cudaMalloc((void **) &d_speed, memsize));

	if (opts.target_type != kGust)
	{
		CUDA_CHECK(cudaMalloc((void **) &d_dir, memsize));
	}

	if (opts.vector_calculation)
	{
		CUDA_CHECK(cudaMalloc((void **) &d_vector, memsize));
	}

	// Copy data to device

	if (opts.u->packed_values)
	{
		// Unpack data and copy it back to host, we need it because its put back to cache
		opts.u->packed_values->Unpack(d_u, &stream);
		CUDA_CHECK(cudaMemcpyAsync(opts.u->values, d_u, memsize, cudaMemcpyDeviceToHost, stream));
	}
	else
	{
		CUDA_CHECK(cudaMemcpyAsync(d_u, opts.u->values, memsize, cudaMemcpyHostToDevice, stream));
	}

	if (opts.v->packed_values)
	{
		// Unpack data and copy it back to host, we need it because its put back to cache
		opts.v->packed_values->Unpack(d_v, &stream);
		CUDA_CHECK(cudaMemcpyAsync(opts.v->values, d_v, memsize, cudaMemcpyDeviceToHost, stream));
	}
	else
	{
		CUDA_CHECK(cudaMemcpyAsync(d_v, opts.v->values, memsize, cudaMemcpyHostToDevice, stream));
	}

	int src=0;

	CUDA_CHECK(cudaMemcpyAsync(d_missing, &src, sizeof(int), cudaMemcpyHostToDevice, stream));

	// dims

	const int blockSize = 128;
	const int gridSize = opts.N/blockSize + (opts.N%blockSize == 0?0:1);

	if (opts.u->south_pole_lat > 0)
	{
		opts.u->south_pole_lat = -opts.u->south_pole_lat;
		opts.u->south_pole_lon = 0;
	}

	/*
	 *  If calculating gust, do not ever rotate grid since we don't calculate
	 * direction for it.
	*/

	CUDA_CHECK(cudaStreamSynchronize(stream));
	
	if (opts.target_type != kGust && opts.need_grid_rotation)
	{
		Rotate <<< gridSize, blockSize, 0, stream >>> (d_u, d_v, *opts.u);
	}
	
	Calculate <<< gridSize, blockSize, 0, stream >>> (d_u, d_v, d_speed, d_dir, d_vector, opts, d_missing);

	// block until the stream has completed
	CUDA_CHECK(cudaStreamSynchronize(stream));

	// check if kernel execution generated an error

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	CUDA_CHECK(cudaMemcpyAsync(opts.speed->values, d_speed, memsize, cudaMemcpyDeviceToHost, stream));

	if (opts.target_type != kGust)
	{
		CUDA_CHECK(cudaMemcpyAsync(opts.dir->values, d_dir, memsize, cudaMemcpyDeviceToHost, stream));
	}

	if (opts.vector_calculation)
	{
		CUDA_CHECK(cudaMemcpyAsync(opts.vector->values, d_vector, memsize, cudaMemcpyDeviceToHost, stream));
	}

	CUDA_CHECK(cudaMemcpyAsync(&opts.missing, d_missing, sizeof(int), cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaStreamSynchronize(stream));

	// Free device memory

	CUDA_CHECK(cudaFree(d_u));
	CUDA_CHECK(cudaFree(d_v));

	CUDA_CHECK(cudaFree(d_speed));
	
	if (d_dir)
	{
		CUDA_CHECK(cudaFree(d_dir));
	}

	if (d_vector)
	{
		CUDA_CHECK(cudaFree(d_vector));
	}
	
	CUDA_CHECK(cudaFree(d_missing));

	CUDA_CHECK(cudaStreamDestroy(stream)); // this blocks
}
