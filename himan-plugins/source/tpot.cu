#include "cuda_plugin_helper.h"
#include "lift.h"

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform.h>

using namespace himan;

template <typename T>
__global__ void ThetaKernel(const T* __restrict__ d_t, const T* __restrict__ d_p, T* __restrict__ d_tp, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		d_tp[idx] = metutil::Theta_<T>(d_t[idx], d_p[idx]);
	}
}

template <typename T>
__global__ void ThetaWKernel(const T* __restrict__ d_tpe, T* __restrict__ d_tpw, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		d_tpw[idx] = metutil::ThetaW_<T>(d_tpe[idx]);
	}
}

template <typename T>
__global__ void ThetaEKernel(const T* __restrict__ d_t, const T* __restrict__ d_p, const T* __restrict__ d_td,
                             T* __restrict__ d_tpe, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		d_tpe[idx] = metutil::ThetaE_<T>(d_t[idx], d_td[idx], d_p[idx]);
	}
}

namespace tpotgpu
{
void Process(std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<info<double>> myTargetInfo, bool theta,
             bool thetaw, bool thetae)
{
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	double* d_t = 0;
	double* d_p = 0;
	double* d_td = 0;
	double* d_tp = 0;
	double* d_tpw = 0;
	double* d_tpe = 0;

	const size_t N = myTargetInfo->SizeLocations();
	const size_t memsize = N * sizeof(double);

	// dims

	const int blockSize = 512;
	const int gridSize = N / blockSize + (N % blockSize == 0 ? 0 : 1);

	info_t TDInfo;

	auto TInfo = cuda::Fetch<double>(conf, myTargetInfo->Time(), myTargetInfo->Level(), param("T-K"),
	                                 myTargetInfo->ForecastType());

	if (!TInfo)
	{
		return;
	}

	if (thetae || thetaw)
	{
		TDInfo = cuda::Fetch<double>(conf, myTargetInfo->Time(), myTargetInfo->Level(), param("TD-K"),
		                             myTargetInfo->ForecastType());

		if (!TDInfo)
		{
			return;
		}
	}

	if (myTargetInfo->Level().Type() != kPressure)
	{
		auto PInfo = cuda::Fetch<double>(conf, myTargetInfo->Time(), myTargetInfo->Level(), param("P-HPA"),
		                                 myTargetInfo->ForecastType());

		if (!PInfo)
		{
			return;
		}

		CUDA_CHECK(cudaMalloc((void**)&d_p, memsize));
		cuda::PrepareInfo(PInfo, d_p, stream, conf->UseCacheForReads());

		if (PInfo->Param().Unit() == kHPa || PInfo->Param().Name() == "P-HPA")
		{
			thrust::device_ptr<double> dt_p = thrust::device_pointer_cast(d_p);
			thrust::transform(thrust::cuda::par.on(stream), dt_p, dt_p + N, dt_p,
			                  [] __device__(double d) { return d * 100; });
		}
	}
	else
	{
		CUDA_CHECK(cudaMalloc((void**)&d_p, memsize));
		thrust::device_ptr<double> dt_p = thrust::device_pointer_cast(d_p);
		thrust::fill_n(thrust::cuda::par.on(stream), dt_p, N, myTargetInfo->Level().Value() * 100);
	}

	CUDA_CHECK(cudaMalloc((void**)&d_t, memsize));
	cuda::PrepareInfo(TInfo, d_t, stream, conf->UseCacheForReads());

	if (TDInfo)
	{
		CUDA_CHECK(cudaMalloc((void**)&d_td, memsize));
		cuda::PrepareInfo(TDInfo, d_td, stream, conf->UseCacheForReads());
	}

	CUDA_CHECK(cudaStreamSynchronize(stream));

	if (theta)
	{
		CUDA_CHECK(cudaMalloc((void**)&d_tp, memsize));
		ThetaKernel<double><<<gridSize, blockSize, 0, stream>>>(d_t, d_p, d_tp, N);
	}
	if (thetae || thetaw)
	{
		CUDA_CHECK(cudaMalloc((void**)&d_tpe, memsize));
		ThetaEKernel<double><<<gridSize, blockSize, 0, stream>>>(d_t, d_p, d_td, d_tpe, N);
	}
	if (thetaw)
	{
		CUDA_CHECK(cudaMalloc((void**)&d_tpw, memsize));
		ThetaWKernel<double><<<gridSize, blockSize, 0, stream>>>(d_tpe, d_tpw, N);
	}

	// block until the device has completed
	CUDA_CHECK(cudaStreamSynchronize(stream));

	// check if kernel execution generated an error

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	// Retrieve result from device

	if (theta)
	{
		myTargetInfo->Find<param>(param("TP-K"));
		cuda::ReleaseInfo(myTargetInfo, d_tp, stream);
	}

	if (thetaw)
	{
		myTargetInfo->Find<param>(param("TPW-K"));
		cuda::ReleaseInfo(myTargetInfo, d_tpw, stream);
	}

	if (thetae)
	{
		myTargetInfo->Find<param>(param("TPE-K"));
		cuda::ReleaseInfo(myTargetInfo, d_tpe, stream);
	}

	CUDA_CHECK(cudaStreamSynchronize(stream));

	if (d_tp)
		CUDA_CHECK(cudaFree(d_tp));
	if (d_tpw)
		CUDA_CHECK(cudaFree(d_tpw));
	if (d_tpe)
		CUDA_CHECK(cudaFree(d_tpe));

	CUDA_CHECK(cudaFree(d_t));
	CUDA_CHECK(cudaFree(d_p));

	if (d_td)
		CUDA_CHECK(cudaFree(d_td));

	CUDA_CHECK(cudaStreamDestroy(stream));
}
}
