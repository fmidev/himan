#include "cuda_plugin_helper.h"
#include "moisture.h"

using namespace himan;

template <typename T>
__global__ void DewpointKernel(const T* __restrict__ d_t, const T* __restrict__ d_rh, T* __restrict__ d_td,
                               double RH_scale, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		ASSERT(d_t[idx] > 80. || IsMissing(d_t[idx]));

		d_td[idx] = metutil::DewPointFromRH_<double>(d_t[idx], min(d_rh[idx] * RH_scale, 100.f));
	}
}

namespace dewpointgpu
{
void Process(std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<info<double>> myTargetInfo)
{
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	const size_t N = myTargetInfo->SizeLocations();
	const size_t memsize = N * sizeof(double);

	// Allocate device arrays

	double* d_t = 0;
	double* d_rh = 0;
	double* d_td = 0;

	auto TInfo = cuda::Fetch<double>(conf, myTargetInfo->Time(), myTargetInfo->Level(), param("T-K"),
	                                 myTargetInfo->ForecastType());
	auto RHInfo = cuda::Fetch<double>(conf, myTargetInfo->Time(), myTargetInfo->Level(),
	                                  {param("RH-PRCNT"), param("RH-0TO1")}, myTargetInfo->ForecastType());

	if (!TInfo || !RHInfo)
	{
		return;
	}

	// Allocate memory on device

	CUDA_CHECK(cudaMalloc((void**)&d_t, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_rh, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_td, memsize));

	cuda::PrepareInfo<double>(TInfo, d_t, stream, conf->UseCacheForReads());
	cuda::PrepareInfo<double>(RHInfo, d_rh, stream, conf->UseCacheForReads());

	double RHScale = 1;

	if (RHInfo->Param().Name() == "RH-0TO1")
	{
		RHScale = 100;
	}

	// dims

	const int blockSize = 512;
	const int gridSize = N / blockSize + (N % blockSize == 0 ? 0 : 1);

	CUDA_CHECK(cudaStreamSynchronize(stream));

	DewpointKernel<double><<<gridSize, blockSize, 0, stream>>>(d_t, d_rh, d_td, RHScale, N);

	cuda::ReleaseInfo<double>(myTargetInfo, d_td, stream);

	CUDA_CHECK(cudaStreamSynchronize(stream));

	// Free device memory

	CUDA_CHECK(cudaFree(d_t));
	CUDA_CHECK(cudaFree(d_td));
	CUDA_CHECK(cudaFree(d_rh));

	CUDA_CHECK(cudaStreamDestroy(stream));
}
}
