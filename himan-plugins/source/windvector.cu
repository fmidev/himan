#include "cuda_plugin_helper.h"
#include "interpolate.h"
#include "plugin_factory.h"
#include "windvector.cuh"

#define HIMAN_AUXILIARY_INCLUDE

#include "cache.h"

#undef HIMAN_AUXILIARY_INCLUDE

/*
 * Calculate results. At this point it as assumed that U and V are in correct form.
 */

__global__ void Calculate(cdarr_t d_u, cdarr_t d_v, darr_t d_speed, darr_t d_dir,
                          himan::plugin::HPWindVectorTargetType targetType, size_t N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		const double U = d_u[idx];
		const double V = d_v[idx];

		d_speed[idx] = __dsqrt_rn(U * U + V * V);

		if (targetType != himan::plugin::kGust)
		{
			int offset = 180;

			if (targetType == himan::plugin::kSea || targetType == himan::plugin::kIce)
			{
				offset = 0;
			}

			double dir = himan::constants::kRad * atan2(U, V) + offset;

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

void himan::plugin::windvector_cuda::RunCuda(std::shared_ptr<const plugin_configuration> conf,
                                             std::shared_ptr<info> myTargetInfo, const param& UParam,
                                             const param& VParam, HPWindVectorTargetType itsTargetType)
{
	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	// Allocate device arrays

	double* d_u = 0;
	double* d_v = 0;
	double* d_speed = 0;
	double* d_dir = 0;

	// Allocate memory on device
	const size_t N = myTargetInfo->SizeLocations();

	const size_t memsize = N * sizeof(double);

	CUDA_CHECK(cudaMalloc((void**)&d_u, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_v, memsize));

	CUDA_CHECK(cudaMalloc((void**)&d_speed, memsize));

	if (itsTargetType != kGust)
	{
		CUDA_CHECK(cudaMalloc((void**)&d_dir, memsize));
	}

	// Fetch U & V, unpack to device, do not copy to host

	auto UInfo = cuda::Fetch(conf, myTargetInfo->Time(), myTargetInfo->Level(), UParam, myTargetInfo->ForecastType());
	auto VInfo = cuda::Fetch(conf, myTargetInfo->Time(), myTargetInfo->Level(), VParam, myTargetInfo->ForecastType());

	if (!UInfo || !VInfo)
	{
		return;
	}

	cuda::Unpack(UInfo, stream, d_u);
	cuda::Unpack(VInfo, stream, d_v);

	// Rotate components; data already at device memory

	if (UInfo->Grid()->UVRelativeToGrid())
	{
		himan::interpolate::RotateVectorComponentsGPU(*UInfo, *VInfo, stream, d_u, d_v);
		CUDA_CHECK(cudaStreamSynchronize(stream));
	}

	// Copy to host

	CUDA_CHECK(cudaMemcpyAsync(UInfo->Data().ValuesAsPOD(), d_u, sizeof(double) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(VInfo->Data().ValuesAsPOD(), d_v, sizeof(double) * N, cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaStreamSynchronize(stream));

	// And finally insert to cache

	auto c = GET_PLUGIN(cache);
	c->Insert(UInfo);
	c->Insert(VInfo);

	for (myTargetInfo->ResetParam(); myTargetInfo->NextParam();)
	{
		myTargetInfo->Grid()->AB(UInfo->Grid()->AB());
	}

	// dims

	const int blockSize = 256;
	const int gridSize = N / blockSize + (N % blockSize == 0 ? 0 : 1);

	Calculate<<<gridSize, blockSize, 0, stream>>>(d_u, d_v, d_speed, d_dir, itsTargetType, N);

	// block until the stream has completed
	CUDA_CHECK(cudaStreamSynchronize(stream));

	// check if kernel execution generated an error

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	myTargetInfo->ParamIndex(0);

	cuda::ReleaseInfo(myTargetInfo, d_speed, stream);

	if (itsTargetType != kGust)
	{
		myTargetInfo->ParamIndex(1);
		cuda::ReleaseInfo(myTargetInfo, d_dir, stream);
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
