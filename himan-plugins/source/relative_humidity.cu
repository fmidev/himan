// System includes

#include "cuda_plugin_helper.h"
#include "moisture.h"

using namespace himan;

// CUDA-kernel that computes RH from T and TD
template <typename Type>
__global__ void CalculateTTD(const Type* __restrict__ d_T, const Type* __restrict__ d_TD, Type* __restrict__ d_RH,
                             size_t N)
{
	const Type b = static_cast<Type>(17.27);
	const Type c = static_cast<Type>(237.3);
	const Type d = static_cast<Type>(1.8);

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		d_RH[idx] = MissingValue<Type>();

		const Type td = d_TD[idx] - static_cast<Type>(constants::kKelvin);
		const Type t = d_T[idx] - static_cast<Type>(constants::kKelvin);
		const Type nomin = exp(d + b * (td / (td + c)));
		const Type denom = exp(d + b * (t / (t + c)));

		const Type RH = nomin / denom;

		if (!IsMissing(RH))
		{
			d_RH[idx] = 100 * max(min(1.0, RH), 0.0);
		}
	}
}

// CUDA-kernel that computes RH from T, Q and P
template <typename Type>
__global__ void CalculateTQP(const Type* __restrict__ d_T, const Type* __restrict__ d_Q, const Type* __restrict__ d_P,
                             Type* __restrict__ d_RH, Type PScale, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		d_RH[idx] = MissingValue<Type>();

		const Type p = d_P[idx] * PScale;
		const Type ES = himan::metutil::Es_<Type>(d_T[idx]) * 0.01;
		const Type RH = (p * d_Q[idx] / constants::kEp / ES) * (p - ES) / (p - d_Q[idx] * p / constants::kEp);

		if (!IsMissing(RH))
		{
			d_RH[idx] = 100.0 * max(min(1.0, RH), 0.0);
		}
	}
}

// CUDA-kernel that computes RH on pressure-level from T and Q
template <typename Type>
__global__ void CalculateTQ(const Type* __restrict__ d_T, const Type* __restrict__ d_Q, Type* __restrict__ d_RH, Type P,
                            size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		d_RH[idx] = MissingValue<Type>();
		const Type ES = himan::metutil::Es_<Type>(d_T[idx]) * 0.01;
		const Type RH = (P * d_Q[idx] / constants::kEp / ES) * (P - ES) / (P - d_Q[idx] * P / constants::kEp);

		if (!IsMissing(RH))
		{
			d_RH[idx] = 100.0 * max(min(1.0, RH), 0.0);
		}
	}
}

void ProcessHumidityGPU(std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<info> myTargetInfo)
{
	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	const size_t N = myTargetInfo->SizeLocations();

	size_t memsize = N * sizeof(float);

	// Define device arrays

	float* d_RH = 0;
	float* d_T = 0;

	// Allocate memory on device

	info_t TInfo =
	    cuda::Fetch(conf, myTargetInfo->Time(), myTargetInfo->Level(), param("T-K"), myTargetInfo->ForecastType());

	if (!TInfo)
	{
		return;
	}

	CUDA_CHECK(cudaMalloc((void**)&d_RH, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_T, memsize));

	cuda::PrepareInfo(TInfo, d_T, stream);

	if (myTargetInfo->Level().Type() == kHybrid)
	{
		const size_t paramIndex = myTargetInfo->ParamIndex();

		for (myTargetInfo->ResetParam(); myTargetInfo->NextParam();)
		{
			myTargetInfo->Grid()->AB(TInfo->Grid()->AB());
		}

		myTargetInfo->ParamIndex(paramIndex);
	}

	// First try to calculate using Q and P

	info_t QInfo =
	    cuda::Fetch(conf, myTargetInfo->Time(), myTargetInfo->Level(), param("Q-KGKG"), myTargetInfo->ForecastType());

	const int blockSize = 512;
	const int gridSize = N / blockSize + (N % blockSize == 0 ? 0 : 1);

	// Select mode in which RH is calculated (with T-TD, T-Q-P or T-Q)
	if (!QInfo)
	{
		// Case where RH is calculated from T and TD
		float* d_TD = 0;

		// Allocate memory on device

		CUDA_CHECK(cudaMalloc((void**)&d_TD, memsize));

		info_t TDInfo =
		    cuda::Fetch(conf, myTargetInfo->Time(), myTargetInfo->Level(), param("TD-K"), myTargetInfo->ForecastType());

		// Copy data to device

		cuda::PrepareInfo(TDInfo, d_TD, stream);

		CalculateTTD<float><<<gridSize, blockSize, 0, stream>>>(d_T, d_TD, d_RH, N);

		// block until the stream has completed
		CUDA_CHECK(cudaStreamSynchronize(stream));

		// check if kernel execution generated an error
		CUDA_CHECK_ERROR_MSG("Kernel invocation");

		CUDA_CHECK(cudaFree(d_TD));
	}
	else if (myTargetInfo->Level().Type() != kPressure)
	{
		info_t PInfo = cuda::Fetch(conf, myTargetInfo->Time(), myTargetInfo->Level(),
		                           params({param("P-HPA"), param("P-PA")}), myTargetInfo->ForecastType());

		if (!PInfo)
		{
			CUDA_CHECK(cudaFree(d_T));
			CUDA_CHECK(cudaFree(d_RH));
			return;
		}

		// Case where RH is calculated from T, Q and P
		float* d_Q = 0;
		float* d_P = 0;

		// Allocate memory on device
		CUDA_CHECK(cudaMalloc((void**)&d_Q, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_P, memsize));

		cuda::PrepareInfo<float>(QInfo, d_Q, stream);
		cuda::PrepareInfo<float>(PInfo, d_P, stream);

		float PScale = 1;

		if (PInfo->Param().Name() == "P-PA")
		{
			PScale = 0.01;
		}

		CalculateTQP<float><<<gridSize, blockSize, 0, stream>>>(d_T, d_Q, d_P, d_RH, PScale, N);

		// block until the stream has completed
		CUDA_CHECK(cudaStreamSynchronize(stream));

		// check if kernel execution generated an error
		CUDA_CHECK_ERROR_MSG("Kernel invocation");

		// Free device memory
		CUDA_CHECK(cudaFree(d_Q));
		CUDA_CHECK(cudaFree(d_P));
	}
	else
	{
		// Case where RH is calculated for pressure levels from T and Q
		float* d_Q = 0;

		// Allocate memory on device
		CUDA_CHECK(cudaMalloc((void**)&d_Q, memsize));

		// Copy data to device
		cuda::PrepareInfo(QInfo, d_Q, stream);

		CalculateTQ<float><<<gridSize, blockSize, 0, stream>>>(d_T, d_Q, d_RH, myTargetInfo->Level().Value(), N);

		// block until the stream has completed
		CUDA_CHECK(cudaStreamSynchronize(stream));

		// check if kernel execution generated an error
		CUDA_CHECK_ERROR_MSG("Kernel invocation");

		// Free device memory

		CUDA_CHECK(cudaFree(d_Q));
	}

	cuda::ReleaseInfo<float>(myTargetInfo, d_RH, stream);
	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaFree(d_T));
	CUDA_CHECK(cudaFree(d_RH));

	cudaStreamDestroy(stream);
}
