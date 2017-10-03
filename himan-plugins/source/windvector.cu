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

	if (idx < opts.N)
	{
		double U = d_u[idx], V = d_v[idx];
		d_speed[idx] = himan::MissingDouble();
		if (d_dir) d_dir[idx] = himan::MissingDouble();

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

			d_dir[idx] = round(dir);
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

	if (d_lon)
	{
		CUDA_CHECK(cudaFree(d_lon));
		CUDA_CHECK(cudaFree(d_lat));
	}

	CUDA_CHECK(cudaStreamDestroy(stream));
}
