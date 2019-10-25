#include "cuda_plugin_helper.h"

using namespace himan;

template <typename Type>
__device__ Type VVMS(Type VV, Type T, Type P)
{
	ASSERT(P < 1200 || IsMissing(P));
	return 287 * -VV * T / (static_cast<Type>(himan::constants::kG) * 100 * P);
}

template <typename T>
__global__ void VVMSKernel(const T* __restrict__ d_t, const T* __restrict__ d_vv, const T* __restrict__ d_p,
                           T* __restrict__ d_vv_ms, T vv_scale, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		d_vv_ms[idx] = vv_scale * VVMS<T>(d_vv[idx], d_t[idx], d_p[idx]);
	}
}

template <typename T>
__global__ void VVMSKernel(const T* __restrict__ d_t, const T* __restrict__ d_vv, const T P, T* __restrict__ d_vv_ms,
                           T vv_scale, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		d_vv_ms[idx] = vv_scale * VVMS<T>(d_vv[idx], d_t[idx], P);
	}
}

namespace vvmsgpu
{
void Process(std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<info<float>> myTargetInfo)
{
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	// Allocate device arrays

	float* d_t = 0;
	float* d_p = 0;
	float* d_vv = 0;
	float* d_vv_ms = 0;

	const size_t N = myTargetInfo->SizeLocations();
	const size_t memsize = N * sizeof(float);

	auto TInfo = cuda::Fetch<float>(conf, myTargetInfo->Time(), myTargetInfo->Level(), param("T-K"),
	                                myTargetInfo->ForecastType());
	auto VVInfo = cuda::Fetch<float>(conf, myTargetInfo->Time(), myTargetInfo->Level(), param("VV-PAS"),
	                                 myTargetInfo->ForecastType());

	if (!TInfo || !VVInfo)
	{
		return;
	}

	CUDA_CHECK(cudaMalloc((void**)&d_vv_ms, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_t, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_vv, memsize));

	cuda::PrepareInfo<float>(TInfo, d_t, stream, conf->UseCacheForReads());
	cuda::PrepareInfo<float>(VVInfo, d_vv, stream, conf->UseCacheForReads());

	// dims

	const int blockSize = 512;
	const int gridSize = N / blockSize + (N % blockSize == 0 ? 0 : 1);

	bool isPressureLevel = (myTargetInfo->Level().Type() == kPressure);

	const float vv_scale = (myTargetInfo->Param().Name() == "VV-MMS") ? 1000. : 1.;

	// "SetAB"

	if (myTargetInfo->Level().Type() == kHybrid)
	{
		const size_t paramIndex = myTargetInfo->Index<param>();

		for (myTargetInfo->Reset<param>(); myTargetInfo->Next<param>();)
		{
			myTargetInfo->Set<level>(TInfo->Level());
		}

		myTargetInfo->Index<param>(paramIndex);
	}

	if (isPressureLevel == false)
	{
		CUDA_CHECK(cudaMalloc((void**)&d_p, memsize));

		auto PInfo = cuda::Fetch<float>(conf, myTargetInfo->Time(), myTargetInfo->Level(), param("P-HPA"),
		                                myTargetInfo->ForecastType());

		if (!PInfo)
		{
			return;
		}
		cuda::PrepareInfo(PInfo, d_p, stream, conf->UseCacheForReads());

		VVMSKernel<float><<<gridSize, blockSize, 0, stream>>>(d_t, d_vv, d_p, d_vv_ms, vv_scale, N);
	}
	else
	{
		VVMSKernel<float>
		    <<<gridSize, blockSize, 0, stream>>>(d_t, d_vv, myTargetInfo->Level().Value(), d_vv_ms, vv_scale, N);
	}

	cuda::ReleaseInfo<float>(myTargetInfo, d_vv_ms, stream);

	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaFree(d_t));
	CUDA_CHECK(cudaFree(d_vv));
	CUDA_CHECK(cudaFree(d_vv_ms));

	if (d_p)
	{
		// himan::ReleaseInfo(opts.p);
		CUDA_CHECK(cudaFree(d_p));
	}

	CUDA_CHECK(cudaStreamDestroy(stream));
}
}
