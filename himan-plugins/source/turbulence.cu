/**
 * @file turbulence.cu
 **/

#include "cuda_plugin_helper.h"
#include "forecast_time.h"
#include "level.h"
#include "util.h"
#include <algorithm>  // std::reverse

using namespace himan;

template <typename T>
__global__ void TurbulenceKernel(const T* __restrict__ d_prevU, const T* __restrict__ d_U,
                                 const T* __restrict__ d_nextU, const T* __restrict__ d_prevV,
                                 const T* __restrict__ d_V, const T* __restrict__ d_nextV,
                                 const T* __restrict__ d_prevH, const T* __restrict__ d_nextH,
                                 const T* __restrict__ d_dX, const T* __restrict__ d_dY, T* __restrict__ d_TI,
                                 T* __restrict__ d_TI2, size_t NX, size_t NY, bool jPositive)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < NX * NY)
	{
		T dUdX;
		T dUdY;
		T dVdX;
		T dVdY;

		// Compute gradients in X
		if (idx % NX == 0)  // left boundary
		{
			dUdX = (d_U[idx + 1] - d_U[idx]) / d_dX[idx / NX];
			dVdX = (d_V[idx + 1] - d_V[idx]) / d_dX[idx / NX];
		}
		else if ((idx + 1) % NX == 0)  // right boundary
		{
			dUdX = (d_U[idx] - d_U[idx - 1]) / d_dX[idx / NX];
			dVdX = (d_V[idx] - d_V[idx - 1]) / d_dX[idx / NX];
		}
		else
		{
			dUdX = (d_U[idx + 1] - d_U[idx - 1]) / (2 * d_dX[idx / NX]);
			dVdX = (d_V[idx + 1] - d_V[idx - 1]) / (2 * d_dX[idx / NX]);
		}

		// Compute gradients in y
		if (idx < NX)  // top boundary
		{
			dUdY = (d_U[idx + NX] - d_U[idx]) / d_dY[idx % NX];
			dVdY = (d_V[idx + NX] - d_V[idx]) / d_dY[idx % NX];
		}
		else if (idx >= NX * (NY - 1))  // bottom boundary
		{
			dUdY = (d_U[idx] - d_U[idx - NX]) / d_dY[idx % NX];
			dVdY = (d_V[idx] - d_V[idx - NX]) / d_dY[idx % NX];
		}
		else
		{
			dUdY = (d_U[idx + NX] - d_U[idx - NX]) / (2 * d_dY[idx % NX]);
			dVdY = (d_V[idx + NX] - d_V[idx - NX]) / (2 * d_dY[idx % NX]);
		}

		// If the grid scans negatively along the j-axis the sign of the gradiend has to be flipped
		if (!jPositive)
		{
			dUdY *= -1;
			dVdY *= -1;
		}

		// Precalculation of wind shear, deformation and convergence
		T WS = sqrt(pow((d_prevU[idx] + d_U[idx] + d_nextU[idx]) / 3, 2) +
		            pow((d_prevV[idx] + d_V[idx] + d_nextV[idx]) / 3, 2));
		T VWS = sqrt(pow((d_nextU[idx] - d_prevU[idx]) / (d_nextH[idx] - d_prevH[idx]), 2) +
		             pow((d_nextV[idx] - d_prevV[idx]) / (d_nextH[idx] - d_prevH[idx]), 2));
		T DEF = sqrt(pow(dUdX - dVdY, 2) + pow(dVdX + dUdY, 2));
		T CVG = -dUdX - dVdY;

		// Calculate scaling factor
		T S;
		T ScaleMax = 40;
		T ScaleMin = 10;

		// todo avoid branch divergen
		if (WS >= ScaleMax)
		{
			S = 1;
		}
		else if (WS >= ScaleMin && WS < ScaleMax)
		{
			S = WS / ScaleMax;
		}
		else
		{
			S = ScaleMin / ScaleMax;
		}

		// Calculation of turbulence indices
		d_TI[idx] = S * VWS * DEF;
		d_TI2[idx] = S * VWS * (DEF + CVG);
	}
}

namespace turbulence_cuda
{
void Process(std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<info<float>> myTargetInfo)
{
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	const size_t N = myTargetInfo->SizeLocations();
	const size_t memsize = N * sizeof(float);

	// Allocate device arrays

	float* d_prevU = 0;
	float* d_U = 0;
	float* d_nextU = 0;

	float* d_prevV = 0;
	float* d_V = 0;
	float* d_nextV = 0;

	float* d_prevH = 0;
	float* d_nextH = 0;

	float* d_dx = 0;
	float* d_dy = 0;

	float* d_TI = 0;
	float* d_TI2 = 0;

	level forecastLevel = myTargetInfo->Level();
	level prevLevel, nextLevel;

	prevLevel = level(myTargetInfo->Level());
	prevLevel.Value(myTargetInfo->Level().Value() - 1);
	prevLevel.Index(prevLevel.Index() - 1);

	nextLevel = level(myTargetInfo->Level());
	nextLevel.Value(myTargetInfo->Level().Value() + 1);
	nextLevel.Index(nextLevel.Index() + 1);

	auto prevUInfo =
	    cuda::Fetch<float>(conf, myTargetInfo->Time(), prevLevel, param("U-MS"), myTargetInfo->ForecastType());
	auto UInfo =
	    cuda::Fetch<float>(conf, myTargetInfo->Time(), forecastLevel, param("U-MS"), myTargetInfo->ForecastType());
	auto nextUInfo =
	    cuda::Fetch<float>(conf, myTargetInfo->Time(), nextLevel, param("U-MS"), myTargetInfo->ForecastType());

	auto prevVInfo =
	    cuda::Fetch<float>(conf, myTargetInfo->Time(), prevLevel, param("V-MS"), myTargetInfo->ForecastType());
	auto VInfo =
	    cuda::Fetch<float>(conf, myTargetInfo->Time(), forecastLevel, param("V-MS"), myTargetInfo->ForecastType());
	auto nextVInfo =
	    cuda::Fetch<float>(conf, myTargetInfo->Time(), nextLevel, param("V-MS"), myTargetInfo->ForecastType());

	auto prevHInfo =
	    cuda::Fetch<float>(conf, myTargetInfo->Time(), prevLevel, param("HL-M"), myTargetInfo->ForecastType());
	auto nextHInfo =
	    cuda::Fetch<float>(conf, myTargetInfo->Time(), nextLevel, param("HL-M"), myTargetInfo->ForecastType());

	if (!prevUInfo || !UInfo || !nextUInfo || !prevVInfo || !VInfo || !nextVInfo || !prevHInfo || !nextHInfo)
	{
		return;
	}

	// Allocate memory on device

	CUDA_CHECK(cudaMalloc((void**)&d_prevU, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_U, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_nextU, memsize));

	CUDA_CHECK(cudaMalloc((void**)&d_prevV, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_V, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_nextV, memsize));

	CUDA_CHECK(cudaMalloc((void**)&d_prevH, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_nextH, memsize));

	CUDA_CHECK(cudaMalloc((void**)&d_TI, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_TI2, memsize));

	cuda::PrepareInfo<float>(prevUInfo, d_prevU, stream, conf->UseCacheForReads());
	cuda::PrepareInfo<float>(UInfo, d_U, stream, conf->UseCacheForReads());
	cuda::PrepareInfo<float>(nextUInfo, d_nextU, stream, conf->UseCacheForReads());

	cuda::PrepareInfo<float>(prevVInfo, d_prevV, stream, conf->UseCacheForReads());
	cuda::PrepareInfo<float>(VInfo, d_V, stream, conf->UseCacheForReads());
	cuda::PrepareInfo<float>(nextVInfo, d_nextV, stream, conf->UseCacheForReads());

	cuda::PrepareInfo<float>(prevHInfo, d_prevH, stream, conf->UseCacheForReads());
	cuda::PrepareInfo<float>(nextHInfo, d_nextH, stream, conf->UseCacheForReads());

	// calculate grid spacing
	//-------------------------------------------------------------------------------
	ASSERT(myTargetInfo->Grid()->Class() == kRegularGrid);

	auto gr = std::dynamic_pointer_cast<regular_grid>(myTargetInfo->Grid());

	const float Di = gr->Di();
	const float Dj = gr->Dj();
	point firstPoint = myTargetInfo->Grid()->FirstPoint();

	const size_t Ni = gr->Ni();
	const size_t Nj = gr->Nj();

	bool jPositive;
	if (gr->ScanningMode() == kTopLeft)
	{
		jPositive = false;
	}
	else if (gr->ScanningMode() == kBottomLeft)
	{
		jPositive = true;
	}
	else
	{
		exit(1);
	}

	std::vector<float> dx, dy;

	switch (UInfo->Grid()->Type())
	{
		case kLambertConformalConic:
		{
			dx = std::vector<float>(Nj, Di);
			dy = std::vector<float>(Ni, Dj);
			break;
		};
		case kRotatedLatitudeLongitude:
			// When working in rotated space, first point must also be rotated
			firstPoint =
			    std::dynamic_pointer_cast<rotated_latitude_longitude_grid>(myTargetInfo->Grid())->Rotate(firstPoint);
#if defined __GNUC__ && __GNUC__ >= 7
			[[fallthrough]];
#else
			// fall-through
#endif
		case kLatitudeLongitude:
		{
			dx = std::vector<float>(Nj, MissingFloat());
			dy = std::vector<float>(Ni, MissingFloat());
			float jPositiveFloat = jPositive ? 1.0f : -1.0f;

			for (size_t i = 0; i < Ni; ++i)
			{
				dy[i] = util::LatitudeLength(0.0f) * Dj / 360.0f;
			}

			for (size_t j = 0; j < Nj; ++j)
			{
				dx[j] = util::LatitudeLength(static_cast<float>(firstPoint.Y()) +
				                             static_cast<float>(j) * Dj * jPositiveFloat) *
				        Di / 360.0f;
			}
			break;
		}
		default:
		{
			himan::Abort();
		}
	}

	if (!jPositive)
		std::reverse(dx.begin(), dx.end());

	CUDA_CHECK(cudaMalloc((void**)&d_dx, Nj * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&d_dy, Ni * sizeof(float)));

	CUDA_CHECK(cudaMemcpyAsync(d_dx, dx.data(), Nj * sizeof(float), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_dy, dy.data(), Ni * sizeof(float), cudaMemcpyHostToDevice, stream));

	//----------------------------------------------------------------------------
	// end grid spacing calculations

	// calculations for plugin
	//----------------------------------------------------------------------------

	const int blockSize = 512;
	const int gridSize = N / blockSize + (N % blockSize == 0 ? 0 : 1);

	TurbulenceKernel<float><<<gridSize, blockSize, 0, stream>>>(d_prevU, d_U, d_nextU, d_prevV, d_V, d_nextV, d_prevH,
	                                                            d_nextH, d_dx, d_dy, d_TI, d_TI2, Ni, Nj, jPositive);

	//----------------------------------------------------------------------------

	myTargetInfo->Index<param>(0);
	cuda::ReleaseInfo<float>(myTargetInfo, d_TI, stream);

	myTargetInfo->Index<param>(1);
	cuda::ReleaseInfo<float>(myTargetInfo, d_TI2, stream);

	// "SetAB"

	if (myTargetInfo->Level().Type() == kHybrid)
	{
		const size_t paramIndex = myTargetInfo->Index<param>();

		for (myTargetInfo->Reset<param>(); myTargetInfo->Next<param>();)
		{
			myTargetInfo->Set<level>(UInfo->Level());
		}

		myTargetInfo->Index<param>(paramIndex);
	}

	CUDA_CHECK(cudaStreamSynchronize(stream));

	// Free device memory

	CUDA_CHECK(cudaFree(d_dx));
	CUDA_CHECK(cudaFree(d_dy));

	CUDA_CHECK(cudaFree(d_prevU));
	CUDA_CHECK(cudaFree(d_U));
	CUDA_CHECK(cudaFree(d_nextU));

	CUDA_CHECK(cudaFree(d_prevV));
	CUDA_CHECK(cudaFree(d_V));
	CUDA_CHECK(cudaFree(d_nextV));

	CUDA_CHECK(cudaFree(d_prevH));
	CUDA_CHECK(cudaFree(d_nextH));

	CUDA_CHECK(cudaFree(d_TI));
	CUDA_CHECK(cudaFree(d_TI2));
	CUDA_CHECK(cudaStreamDestroy(stream));
}
}  // namespace turbulence_cuda
