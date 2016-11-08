// System includes
#include <iostream>
#include <string>

#include "cuda_plugin_helper.h"
#include "windvector.cuh"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

/*
 * Calculate results. At this point it as assumed that U and V are in correct form.
 */

__global__ void Calculate(cdarr_t d_u, cdarr_t d_v, darr_t d_speed, darr_t d_dir,
                          himan::plugin::windvector_cuda::options opts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	using himan::kFloatMissing;

	if (idx < opts.N)
	{
		double U = d_u[idx], V = d_v[idx];
		d_speed[idx] = kFloatMissing;
		if (d_dir) d_dir[idx] = kFloatMissing;

		if (U != kFloatMissing && V != kFloatMissing)
		{
			double speed = sqrt(U * U + V * V);

			d_speed[idx] = speed;

			double dir = 0;

			if (opts.target_type != himan::plugin::kGust)
			{
				int offset = 180;

				if (opts.target_type == himan::plugin::kSea || opts.target_type == himan::plugin::kIce)
				{
					offset = 0;
				}

				if (speed > 0)
				{
					dir = himan::constants::kRad * atan2(U, V) + offset;

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
		}
	}
}

/*
 * Rotate U and V vectors that are grid-relative (U is pointing to grid north, V to grid east) to earth
 * relative form (U points to earth or map north etc).
 */

__global__ void RotateLambert(double* __restrict__ d_u, double* __restrict__ d_v, const double* __restrict__ d_lon,
                              const double* __restrict__ d_lat, double cone, double orientation,
                              himan::info_simple opts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.size_x * opts.size_y)
	{
		double U = d_u[idx];
		double V = d_v[idx];

		if (U != himan::kFloatMissing && V != himan::kFloatMissing)
		{
			int i = fmod(static_cast<double>(idx), static_cast<double>(opts.size_x));  // idx - j * opts.size_x;
			int j = floor(static_cast<double>(idx / opts.size_x));

			double londiff = d_lon[idx] - orientation;
			const double angle = cone * londiff * himan::constants::kDeg;
			double sinx, cosx;
			sincos(angle, &sinx, &cosx);
			d_u[idx] = -1 * cosx * U + sinx * V;
			d_v[idx] = -1 * -sinx * U + cosx * V;
		}
	}
}

__global__ void Rotate(double* __restrict__ d_u, double* __restrict__ d_v, himan::info_simple opts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.size_x * opts.size_y)
	{
		double U = d_u[idx];
		double V = d_v[idx];

		if (U != himan::kFloatMissing && V != himan::kFloatMissing)
		{
			int i = fmod(static_cast<double>(idx), static_cast<double>(opts.size_x));  // idx - j * opts.size_x;
			int j = floor(static_cast<double>(idx / opts.size_x));

			double lon = opts.first_lon + i * opts.di;

			double lat = himan::kFloatMissing;

			if (opts.j_scans_positive)
			{
				lat = opts.first_lat + j * opts.dj;
			}
			else
			{
				lat = opts.first_lat - j * opts.dj;
			}

			double SinYPole = sin((opts.south_pole_lat + 90.) * himan::constants::kDeg);
			double CosYPole = cos((opts.south_pole_lat + 90.) * himan::constants::kDeg);

			double SinXRot, CosXRot, SinYRot, CosYRot;

			sincos(lon * himan::constants::kDeg, &SinXRot, &CosXRot);
			sincos(lat * himan::constants::kDeg, &SinYRot, &CosYRot);

			double SinYReg = CosYPole * SinYRot + SinYPole * CosYRot * CosXRot;

			SinYReg = MIN(MAX(SinYReg, -1.), 1.);

			double YReg = asin(SinYReg) * himan::constants::kRad;

			double CosYReg = cos(YReg * himan::constants::kDeg);

			double CosXReg = (CosYPole * CosYRot * CosXRot - SinYPole * SinYRot) / CosYReg;

			CosXReg = MIN(MAX(CosXReg, -1.), 1.);
			double SinXReg = CosYRot * SinXRot / CosYReg;

			double XReg = acos(CosXReg) * himan::constants::kRad;

			if (SinXReg < 0.)
			{
				XReg = -XReg;
			}

			XReg += opts.south_pole_lon;

			// UV to earth relative

			double zxmxc = himan::constants::kDeg * (XReg - opts.south_pole_lon);

			double sinxmxc, cosxmxc;

			sincos(zxmxc, &sinxmxc, &cosxmxc);

			double PA = cosxmxc * CosXRot + CosYPole * sinxmxc * SinXRot;
			double PB =
			    CosYPole * sinxmxc * CosXRot * SinYRot + SinYPole * sinxmxc * CosYRot - cosxmxc * SinXRot * SinYRot;
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
	double* d_lon = 0;
	double* d_lat = 0;

	// Allocate memory on device

	size_t memsize = opts.N * sizeof(double);

	CUDA_CHECK(cudaMalloc((void**)&d_u, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_v, memsize));

	CUDA_CHECK(cudaMalloc((void**)&d_speed, memsize));

	if (opts.target_type != kGust)
	{
		CUDA_CHECK(cudaMalloc((void**)&d_dir, memsize));
		PrepareInfo(opts.dir);
	}

	// Copy data to device

	PrepareInfo(opts.u, d_u, stream);
	PrepareInfo(opts.v, d_v, stream);
	PrepareInfo(opts.speed);

	// dims

	const int blockSize = 256;
	const int gridSize = opts.N / blockSize + (opts.N % blockSize == 0 ? 0 : 1);

	/*
	 *  If calculating gust, do not ever rotate grid since we don't calculate
	 * direction for it.
	*/

	CUDA_CHECK(cudaStreamSynchronize(stream));

	if (opts.target_type != kGust && opts.need_grid_rotation)
	{
		if (opts.u->projection == kRotatedLatitudeLongitude)
		{
			if (opts.u->south_pole_lat > 0)
			{
				opts.u->south_pole_lat = -opts.u->south_pole_lat;
				opts.u->south_pole_lon = 0;
			}

			Rotate<<<gridSize, blockSize, 0, stream>>>(d_u, d_v, *opts.u);
		}
		else if (opts.u->projection == kLambertConformalConic)
		{
			const double latin1 = opts.u->latin1;
			const double latin2 = opts.u->latin2;
			assert(latin1 != kFloatMissing);
			double cone;
			if (latin1 == latin2 && latin2 != kFloatMissing)
			{
				cone = sin(latin1 * constants::kDeg);
			}
			else
			{
				cone = (log(cos(latin1 * constants::kDeg)) - log(cos(latin2 * constants::kDeg))) /
				       (log(tan((90 - fabs(latin1)) * constants::kDeg * 0.5)) -
				        log(tan(90 - fabs(latin2)) * constants::kDeg * 0.5));
			}

			CUDA_CHECK(cudaMalloc((void**)&d_lon, memsize));
			CUDA_CHECK(cudaMalloc((void**)&d_lat, memsize));

			CUDA_CHECK(cudaMemcpyAsync(d_lon, opts.lon, memsize, cudaMemcpyHostToDevice, stream));
			CUDA_CHECK(cudaMemcpyAsync(d_lat, opts.lat, memsize, cudaMemcpyHostToDevice, stream));

			RotateLambert<<<gridSize, blockSize, 0, stream>>>(d_u, d_v, d_lon, d_lat, cone, opts.u->orientation,
			                                                  *opts.u);
		}
	}

	Calculate<<<gridSize, blockSize, 0, stream>>>(d_u, d_v, d_speed, d_dir, opts);

	// block until the stream has completed
	CUDA_CHECK(cudaStreamSynchronize(stream));

	// check if kernel execution generated an error

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	ReleaseInfo(opts.u);
	ReleaseInfo(opts.v);
	ReleaseInfo(opts.speed, d_speed, stream);

	if (opts.target_type != kGust)
	{
		ReleaseInfo(opts.dir, d_dir, stream);
	}

	CUDA_CHECK(cudaStreamSynchronize(stream));

	// Free device memory

	CUDA_CHECK(cudaFree(d_u));
	CUDA_CHECK(cudaFree(d_v));
	CUDA_CHECK(cudaFree(d_speed));

	if (d_dir)
	{
		CUDA_CHECK(cudaFree(d_dir));
	}

	CUDA_CHECK(cudaStreamDestroy(stream));
}
