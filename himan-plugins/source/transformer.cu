#include "cuda_plugin_helper.h"

using namespace himan;

template <typename T>
__global__ void TransformerKernel(const T* __restrict__ d_source, T* __restrict__ d_dest, double scale, double base,
                                  size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		d_dest[idx] = __fma_rn(d_source[idx], scale, base);
	}
}

namespace transformergpu
{
void Process(std::shared_ptr<const himan::plugin_configuration> conf, info_t myTargetInfo, info_t sourceInfo,
             double scale, double base)
{
	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	const size_t N = myTargetInfo->SizeLocations();
	size_t memsize = N * sizeof(double);

	// Allocate device arrays

	double *d_source = 0, *d_dest = 0;

	// Allocate memory on device

	CUDA_CHECK(cudaMalloc((void**)&d_source, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_dest, memsize));

	// Copy data to device

	cuda::PrepareInfo(sourceInfo, d_source, stream, conf->UseCacheForReads());

	// dims

	const int blockSize = 512;
	const int gridSize = N / blockSize + (N % blockSize == 0 ? 0 : 1);

	CUDA_CHECK(cudaStreamSynchronize(stream));

	TransformerKernel<double><<<gridSize, blockSize, 0, stream>>>(d_source, d_dest, scale, base, N);
	cuda::ReleaseInfo(myTargetInfo, d_dest, stream);

	// block until the stream has completed
	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaFree(d_source));
	CUDA_CHECK(cudaFree(d_dest));

	cudaStreamDestroy(stream);
}
}
