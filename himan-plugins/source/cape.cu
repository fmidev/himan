// System includes
#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/execution_policy.h>

#include "plugin_factory.h"

#include "cape.cuh"
#include "cuda_helper.h"
#include "lift.h"
#include "util.h"

#include <NFmiGribPacking.h>

#include "forecast_time.h"
#include "level.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "cache.h"
#include "fetcher.h"
#include "hitool.h"

#include "debug.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan;
using namespace himan::numerical_functions;
using namespace himan::plugin;

himan::level cape_cuda::itsBottomLevel;
bool cape_cuda::itsUseVirtualTemperature;

extern float Max(const std::vector<float>& vec);
extern std::vector<float> Convert(const std::vector<double>& arr);
extern std::vector<double> Convert(const std::vector<float>& arr);

template <typename T>
__global__ void InitializeArrayKernel(T* d_arr, T val, size_t N)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (; idx < N; idx += stride)
	{
		d_arr[idx] = val;
	}
}

template <typename T>
void InitializeArray(T* d_arr, T val, size_t N, cudaStream_t& stream)
{
	const int blockSize = 128;
	const int gridSize = N / blockSize + (N % blockSize == 0 ? 0 : 1);

	InitializeArrayKernel<T><<<gridSize, blockSize, 0, stream>>>(d_arr, val, N);
}

template <typename T>
__global__ void MultiplyWith(T* d_arr, T val, size_t N)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (; idx < N; idx += stride)
	{
		d_arr[idx] = d_arr[idx] * val;
	}
}

template <typename T>
void MultiplyWith(T* d_arr, T val, size_t N, cudaStream_t& stream)
{
	const int blockSize = 128;
	const int gridSize = N / blockSize + (N % blockSize == 0 ? 0 : 1);

	MultiplyWith<T><<<gridSize, blockSize, 0, stream>>>(d_arr, val, N);
}

void PrepareInfo(std::shared_ptr<himan::info> fullInfo, cudaStream_t& stream, float* d_farr)
{
	const size_t N = fullInfo->SizeLocations();

	ASSERT(N > 0);
	ASSERT(d_farr);

	// 1. Reserve memory at device for unpacked data
	double* d_arr = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<double**>(&d_arr), N * sizeof(double)));

	// 2. Unpack if needed, leave data to device and simultaneously copy it back to cpu (himan cache)
	auto tempGrid = fullInfo->Grid();

	if (tempGrid->IsPackedData())
	{
		ASSERT(tempGrid->PackedData().ClassName() == "simple_packed" ||
		       tempGrid->PackedData().ClassName() == "jpeg_packed");
		ASSERT(N > 0);
		ASSERT(tempGrid->Data().Size() == N);

		double* arr = const_cast<double*>(tempGrid->Data().ValuesAsPOD());
		CUDA_CHECK(cudaHostRegister(reinterpret_cast<void*>(arr), sizeof(double) * N, 0));

		ASSERT(arr);

		tempGrid->PackedData().Unpack(d_arr, N, &stream);

		CUDA_CHECK(cudaMemcpyAsync(arr, d_arr, sizeof(double) * N, cudaMemcpyDeviceToHost, stream));

		tempGrid->PackedData().Clear();

		auto c = GET_PLUGIN(cache);

		CUDA_CHECK(cudaStreamSynchronize(stream));

		c->Insert(fullInfo);

		CUDA_CHECK(cudaHostUnregister(arr));
	}
	else
	{
		CUDA_CHECK(
		    cudaMemcpyAsync(d_arr, fullInfo->Data().ValuesAsPOD(), sizeof(double) * N, cudaMemcpyHostToDevice, stream));
	}

	thrust::device_ptr<double> dt_arr = thrust::device_pointer_cast(d_arr);
	thrust::device_ptr<float> dt_farr = thrust::device_pointer_cast(d_farr);

	thrust::copy(thrust::cuda::par.on(stream), dt_arr, dt_arr + N, dt_farr);
	thrust::replace_if(thrust::cuda::par.on(stream), dt_farr, dt_farr + N,
	                   [] __device__(const float& val) { return ::isnan(val); }, himan::MissingFloat());

	CUDA_CHECK(cudaStreamSynchronize(stream));
	CUDA_CHECK(cudaFree(d_arr));
}

std::shared_ptr<himan::info> Fetch(const std::shared_ptr<const plugin_configuration> conf,
                                   const himan::forecast_time& theTime, const himan::level& theLevel,
                                   const himan::param& theParam, const himan::forecast_type& theType,
                                   bool returnPacked = true)
{
	try
	{
		auto f = GET_PLUGIN(fetcher);
		return f->Fetch(conf, theTime, theLevel, theParam, theType, returnPacked);
	}
	catch (HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw std::runtime_error("cape_cuda::Fetch(): Unable to proceed");
		}

		return std::shared_ptr<info>();
	}
}

__global__ void CapELValuesKernel(const float* __restrict__ d_CAPE, float* __restrict__ d_ELT,
                                  float* __restrict__ d_ELP, float* __restrict__ d_ELZ, float* __restrict__ d_LastELT,
                                  float* __restrict__ d_LastELP, float* __restrict__ d_LastELZ,
                                  const float* __restrict__ d_Tenv, const float* __restrict__ d_Penv,
                                  const float* __restrict__ d_Zenv, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		// If the CAPE area is continued all the way to stopLevel and beyond, we don't have an EL for that
		// (since integration is forcefully stopped)
		// In this case let last level be EL

		if (d_CAPE[idx] > 0 && IsMissing(d_ELT[idx]))
		{
			d_ELT[idx] = d_Tenv[idx];
			d_ELP[idx] = d_Penv[idx];
			d_ELZ[idx] = d_Zenv[idx];

			d_LastELT[idx] = d_Tenv[idx];
			d_LastELP[idx] = d_Penv[idx];
			d_LastELZ[idx] = d_Zenv[idx];
		}
	}
}

__global__ void VirtualTemperatureKernel(float* __restrict__ d_T, const float* __restrict__ d_P, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		d_T[idx] = himan::metutil::VirtualTemperature_<float>(d_T[idx], d_P[idx] * 100);
	}
}

__global__ void CopyLFCIteratorValuesKernel(float* __restrict__ d_Titer, const float* __restrict__ d_Tparcel,
                                            float* __restrict__ d_Piter, const float* __restrict__ d_Penv, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		if (!IsMissing(d_Tparcel[idx]) && !IsMissing(d_Penv[idx]))
		{
			d_Titer[idx] = d_Tparcel[idx];
			d_Piter[idx] = d_Penv[idx];
		}
	}
}

__global__ void LiftLCLKernel(const float* __restrict__ d_P, const float* __restrict__ d_T,
                              const float* __restrict__ d_PLCL, const float* __restrict__ d_Ptarget,
                              float* __restrict__ d_Tparcel, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		ASSERT((d_P[idx] > 10 && d_P[idx] < 1500) || IsMissing(d_P[idx]));
		ASSERT((d_Ptarget[idx] > 10 && d_Ptarget[idx] < 1500) || IsMissing(d_Ptarget[idx]));
		ASSERT((d_T[idx] > 100 && d_T[idx] < 350) || IsMissing(d_T[idx]));

		const float T = metutil::LiftLCLA_<float>(d_P[idx] * 100, d_T[idx], d_PLCL[idx] * 100, d_Ptarget[idx] * 100);

		ASSERT((T > 100 && T < 350) || IsMissing(T));

		d_Tparcel[idx] = T;
	}
}

__global__ void MoistLiftKernel(const float* __restrict__ d_T, const float* __restrict__ d_P,
                                const float* __restrict__ d_Ptarget, float* __restrict__ d_Tparcel, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	ASSERT(d_T);
	ASSERT(d_P);

	if (idx < N)
	{
		ASSERT((d_P[idx] > 10 && d_P[idx] < 1500) || IsMissing(d_P[idx]));
		ASSERT((d_Ptarget[idx] > 10 && d_Ptarget[idx] < 1500) || IsMissing(d_Ptarget[idx]));

		ASSERT((d_T[idx] > 100 && d_T[idx] < 350) || IsMissing(d_T[idx]));

		float T = metutil::MoistLiftA_<float>(d_P[idx] * 100, d_T[idx], d_Ptarget[idx] * 100);

		ASSERT((T > 100 && T < 350) || IsMissing(T));

		d_Tparcel[idx] = T;
	}
}

__global__ void CAPEKernel(const float* __restrict__ d_Tenv, const float* __restrict__ d_Penv,
                           const float* __restrict__ d_Zenv, const float* __restrict__ d_prevTenv,
                           const float* __restrict__ d_prevPenv, const float* __restrict__ d_prevZenv,
                           const float* __restrict__ d_Tparcel, const float* __restrict__ d_prevTparcel,
                           const float* __restrict__ d_LFCT, const float* __restrict__ d_LFCP,
                           float* __restrict__ d_CAPE, float* __restrict__ d_CAPE1040, float* __restrict__ d_CAPE3km,
                           float* __restrict__ d_ELT, float* __restrict__ d_ELP, float* __restrict__ d_ELZ,
                           float* __restrict__ d_LastELT, float* __restrict__ d_LastELP, float* __restrict__ d_LastELZ,
                           unsigned char* __restrict__ d_found, int d_curLevel, int d_breakLevel, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N && d_found[idx] == 0)
	{
		float Tenv = d_Tenv[idx];
		ASSERT(Tenv > 100.);

		float Penv = d_Penv[idx];  // hPa
		ASSERT(Penv < 1200.);

		float Zenv = d_Zenv[idx];  // m

		float prevTenv = d_prevTenv[idx];  // K
		ASSERT(prevTenv > 100.);

		float prevPenv = d_prevPenv[idx];  // hPa
		ASSERT(prevPenv < 1200.);

		float prevZenv = d_prevZenv[idx];  // m

		float Tparcel = d_Tparcel[idx];  // K
		ASSERT(Tparcel > 100. || IsMissing(Tparcel));

		float prevTparcel = d_prevTparcel[idx];  // K
		ASSERT(prevTparcel > 100. || IsMissing(prevTparcel));

		const float LFCP = d_LFCP[idx];  // hPa
		const float LFCT = d_LFCT[idx];  // K

		if (IsMissing(Penv) || IsMissing(Tenv) || IsMissing(Zenv) || IsMissing(prevZenv) || IsMissing(Tparcel) ||
		    Penv > LFCP)
		{
			// Missing data or current grid point is below LFC
			return;
		}

		ASSERT(LFCP < 1200.);
		ASSERT(LFCT > 100.);

		if (IsMissing(prevTparcel) && !IsMissing(Tparcel))
		{
			// When rising above LFC, get accurate value of Tenv at that level so that even small amounts of CAPE
			// (and EL!) values can be determined.

			prevTenv = interpolation::Linear<float>(LFCP, prevPenv, Penv, prevTenv, Tenv);
			prevZenv = interpolation::Linear<float>(LFCP, prevPenv, Penv, prevZenv, Zenv);
			prevPenv = LFCP;     // LFC pressure
			prevTparcel = LFCT;  // LFC temperature

			// If LFC was found close to lower hybrid level, the linear interpolation and moist lift will result
			// to same values. In this case CAPE integration fails as there is no area formed between environment
			// and parcel temperature. The result for this is that LFC is found but EL is not found. To prevent
			// this, warm the parcel value just slightly so that a miniscule CAPE area is formed and EL is found.

			if (fabs(prevTparcel - prevTenv) < 0.0001)
			{
				prevTparcel += 0.0001;
			}
		}

		if (d_curLevel < d_breakLevel && (Tenv - Tparcel) > 25.)
		{
			// Temperature gap between environment and parcel too large --> abort search.
			// Only for values higher in the atmosphere, to avoid the effects of inversion

			d_found[idx] = 1;
		}
		else
		{
			if (prevZenv < 3000.)
			{
				float C = CAPE::CalcCAPE3km(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv);

				d_CAPE3km[idx] += C;

				ASSERT(d_CAPE3km[idx] >= 0);
			}

			float C = CAPE::CalcCAPE1040(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv);

			d_CAPE1040[idx] += C;

			ASSERT(d_CAPE1040[idx] >= 0);

			float CAPE, ELT, ELP, ELZ;
			CAPE::CalcCAPE(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv, CAPE, ELT, ELP, ELZ);

			d_CAPE[idx] += CAPE;

			ASSERT(CAPE >= 0.);

			if (IsValid(ELT))
			{
				if (IsMissing(d_ELT[idx]))
				{
					d_ELT[idx] = ELT;
				}

				d_ELP[idx] = fmaxf(ELP, d_ELP[idx]);
				d_ELZ[idx] = fminf(ELZ, d_ELZ[idx]);

				d_LastELT[idx] = ELT;
				d_LastELP[idx] = ELP;
				d_LastELZ[idx] = ELZ;
			}
		}
	}
}

__global__ void CINKernel(const float* __restrict__ d_Tenv, const float* __restrict__ d_prevTenv,
                          const float* __restrict__ d_Penv, const float* __restrict__ d_prevPenv,
                          const float* __restrict__ d_Zenv, const float* __restrict__ d_prevZenv,
                          const float* __restrict__ d_Tparcel, const float* __restrict__ d_prevTparcel,
                          const float* __restrict__ d_PLCL, const float* __restrict__ d_PLFC,
                          const float* __restrict__ d_Psource, float* __restrict__ d_cinh,
                          unsigned char* __restrict__ d_found, bool useVirtualTemperature, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N && d_found[idx] == 0)
	{
		float Tenv = d_Tenv[idx];  // K
		ASSERT(Tenv >= 150.);

		const float prevTenv = d_prevTenv[idx];

		float Penv = d_Penv[idx];  // hPa
		ASSERT(Penv < 1200. || IsMissing(Penv));

		const float prevPenv = d_prevPenv[idx];

		float Tparcel = d_Tparcel[idx];  // K
		ASSERT(Tparcel >= 150. || IsMissing(Tparcel));

		const float prevTparcel = d_prevTparcel[idx];

		const float PLFC = d_PLFC[idx];  // hPa
		ASSERT(PLFC < 1200. || IsMissing(PLFC));

		const float PLCL = d_PLCL[idx];  // hPa
		ASSERT(PLCL < 1200. || IsMissing(PLCL));

		float Zenv = d_Zenv[idx];          // m
		float prevZenv = d_prevZenv[idx];  // m

		// Make sure we have passed the starting level
		if (Penv <= d_Psource[idx])
		{
			if (Penv <= PLFC)
			{
				// reached max height
				d_found[idx] = 1;

				// Integrate the final piece from previous level to LFC level

				if (IsMissing(prevTparcel) || IsMissing(prevPenv) || IsMissing(prevTenv))
				{
					Tparcel = MissingFloat();  // unable to proceed with CIN integration
				}
				else
				{
					// First get LFC height in meters
					Zenv = interpolation::Linear<float>(PLFC, prevPenv, Penv, prevZenv, Zenv);

					// LFC environment temperature value
					Tenv = interpolation::Linear<float>(PLFC, prevPenv, Penv, prevTenv, Tenv);

					// LFC T parcel value
					Tparcel = interpolation::Linear<float>(PLFC, prevPenv, Penv, prevTparcel, Tparcel);

					Penv = PLFC;

					if (Zenv < prevZenv)
					{
						prevZenv = Zenv;
					}
				}
			}

			if (Penv < PLCL && useVirtualTemperature)
			{
				// Above LCL, switch to virtual temperature

				Tparcel = metutil::VirtualTemperature_<float>(Tparcel, Penv * 100);
				Tenv = metutil::VirtualTemperature_<float>(Tenv, Penv * 100);
			}

			if (!IsMissing(Tparcel))
			{
				d_cinh[idx] += CAPE::CalcCIN(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv);
				ASSERT(d_cinh[idx] <= 0);
			}
		}
	}
}

__global__ void LFCKernel(const float* __restrict__ d_T, const float* __restrict__ d_P,
                          const float* __restrict__ d_prevT, const float* __restrict__ d_prevP,
                          float* __restrict__ d_Tparcel, const float* __restrict__ d_prevTparcel,
                          const float* __restrict__ d_LCLT, const float* __restrict__ d_LCLP,
                          float* __restrict__ d_LFCT, float* __restrict__ d_LFCP, unsigned char* __restrict__ d_found,
                          int d_curLevel, int d_breakLevel, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N && d_found[idx] == 0)
	{
		float Tparcel = d_Tparcel[idx];
		float prevTparcel = d_prevTparcel[idx];
		float Tenv = d_T[idx];

		ASSERT(Tenv < 350.);
		ASSERT(Tenv > 100.);

		float prevTenv = d_prevT[idx];
		ASSERT(prevTenv < 350.);
		ASSERT(prevTenv > 100.);

		float Penv = d_P[idx];
		float prevPenv = d_prevP[idx];

		ASSERT(Penv > 50.);
		ASSERT(Penv < 1200.);
		float LCLP = d_LCLP[idx];
		ASSERT(prevPenv > 50.);
		ASSERT(prevPenv < 1200.);

		if (d_curLevel < d_breakLevel && (Tenv - Tparcel) > 30.)
		{
			// Temperature gap between environment and parcel too large --> abort search.
			// Only for values higher in the atmosphere, to avoid the effects of inversion

			d_found[idx] = 1;
		}

		const float diff = Tparcel - Tenv;

		if (Penv < LCLP && diff > 0)
		{
			d_found[idx] = 1;

			if (IsMissing(prevTparcel))
			{
				prevTparcel = d_LCLT[idx];  // previous is LCL
				ASSERT(!IsMissing(d_LCLT[idx]));
			}

			if (diff < 0.1)
			{
				d_LFCT[idx] = Tparcel;
				d_LFCP[idx] = Penv;
			}
			else if (prevTparcel - prevTenv >= 0)
			{
				d_LFCT[idx] = prevTparcel;
				d_LFCP[idx] = prevPenv;
			}
			else
			{
				auto intersection = CAPE::GetPointOfIntersection(point(Tenv, Penv), point(prevTenv, prevPenv),
				                                                 point(Tparcel, Penv), point(prevTparcel, prevPenv));

				d_LFCT[idx] = intersection.X();
				d_LFCP[idx] = intersection.Y();

				if (d_LFCP[idx] > prevPenv)
				{
					// Do not allow LFC to be below previous level; if intersection fails to put it in the correct
					// "bin" (between previous and current pressure), use the only information that certain:
					// the crossing has happened at least at current pressure
					d_LFCT[idx] = Tparcel;
					d_LFCP[idx] = Penv;
				}
				else if (IsMissing(d_LFCT[idx]))
				{
					// Intersection not found, use exact level value
					d_LFCT[idx] = Tparcel;
					d_LFCP[idx] = Penv;
				}
			}

			ASSERT(d_LFCT[idx] > 100);
			ASSERT(d_LFCT[idx] < 350);
		}
	}
}

__global__ void ThetaEKernel(const float* __restrict__ d_T, const float* __restrict__ d_RH,
                             const float* __restrict__ d_P, const float* __restrict__ d_prevT,
                             const float* __restrict__ d_prevRH, const float* __restrict__ d_prevP,
                             float* __restrict__ d_maxThetaE, float* __restrict__ d_Tresult,
                             float* __restrict__ d_TDresult, float* __restrict__ d_Presult,
                             unsigned char* __restrict__ d_found, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N && d_found[idx] == 0)
	{
		float T = d_T[idx];
		float P = d_P[idx];
		float RH = d_RH[idx];

		if (P < 600.)
		{
			// Cut search if reach level 600hPa

			// Linearly interpolate temperature and humidity values to 600hPa, to check
			// if highest theta e is found there

			T = interpolation::Linear<float>(600.f, P, d_prevP[idx], T, d_prevT[idx]);
			RH = interpolation::Linear<float>(600.f, P, d_prevP[idx], RH, d_prevRH[idx]);

			d_found[idx] = 1;  // Make sure this is the last time we access this grid point
			P = 600.;
		}

		float TD = metutil::DewPointFromRH_<float>(T, RH);

		float& refThetaE = d_maxThetaE[idx];
		float ThetaE = metutil::smarttool::ThetaE_<float>(T, RH, P * 100);

		if ((ThetaE - refThetaE) > 0.0001)  // epsilon added for numerical stability
		{
			refThetaE = ThetaE;
			d_Tresult[idx] = T;
			d_TDresult[idx] = TD;
			d_Presult[idx] = P;
		}
	}
}

__global__ void MixingRatioKernel(const float* __restrict__ d_T, float* __restrict__ d_P,
                                  const float* __restrict__ d_RH, float* __restrict__ d_Tpot, float* __restrict__ d_MR,
                                  size_t N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	ASSERT(d_T);
	ASSERT(d_RH);
	ASSERT(d_P);

	if (idx < N)
	{
		const float T = d_T[idx];
		const float P = d_P[idx];
		const float RH = d_RH[idx];

		ASSERT((T > 150 && T < 350) || IsMissing(T));
		ASSERT((P > 100 && P < 1500) || IsMissing(P));
		ASSERT((RH >= 0 && RH < 102) || IsMissing(RH));

		d_Tpot[idx] = metutil::Theta_<float>(T, 100 * P);
		d_MR[idx] = metutil::smarttool::MixingRatio_<float>(T, RH, 100 * P);

		d_P[idx] = P - 2.0;
	}
}

__global__ void MixingRatioFinalizeKernel(float* __restrict__ d_T, float* __restrict__ d_TD,
                                          const float* __restrict__ d_P, const float* __restrict__ d_Tpot,
                                          const float* __restrict__ d_MR, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		const float P = d_P[idx];
		const float MR = d_MR[idx];
		const float Tpot = d_Tpot[idx];

		ASSERT((P > 100 && P < 1500) || IsMissing(P));

		const float T = Tpot * pow((P / 1000.), 0.2854);
		const float Es = metutil::Es_<float>(T);  // Saturated water vapor pressure
		const float E = metutil::E_<float>(MR, 100 * P);
		const float RH = E / Es * 100;

		d_TD[idx] = metutil::DewPointFromRH_<float>(T, RH);
		d_T[idx] = T;
	}
}

cape_source cape_cuda::GetHighestThetaEValuesGPU(const std::shared_ptr<const plugin_configuration> conf,
                                                 std::shared_ptr<info> myTargetInfo)
{
	himan::level curLevel = itsBottomLevel;

	const size_t N = myTargetInfo->Data().Size();
	const int blockSize = 256;
	const int gridSize = N / blockSize + (N % blockSize == 0 ? 0 : 1);

	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	float* d_maxThetaE = 0;
	float* d_Tresult = 0;
	float* d_TDresult = 0;
	float* d_Presult = 0;
	float* d_T = 0;
	float* d_P = 0;
	float* d_RH = 0;
	float* d_prevT = 0;
	float* d_prevP = 0;
	float* d_prevRH = 0;

	unsigned char* d_found = 0;

	CUDA_CHECK(cudaMalloc((float**)&d_maxThetaE, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_Tresult, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_TDresult, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_Presult, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_found, sizeof(unsigned char) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_T, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_P, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_RH, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_prevT, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_prevP, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_prevRH, sizeof(float) * N));

	InitializeArray<float>(d_maxThetaE, -1, N, stream);
	InitializeArray<float>(d_Tresult, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_TDresult, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_Presult, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_prevT, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_prevP, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_prevRH, himan::MissingFloat(), N, stream);

	InitializeArray<unsigned char>(d_found, 0, N, stream);

	thrust::device_ptr<unsigned char> dt_found = thrust::device_pointer_cast(d_found);

	while (true)
	{
		auto TInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType());
		auto RHInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("RH-PRCNT"), myTargetInfo->ForecastType());
		auto PInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType());

		if (!TInfo || !RHInfo || !PInfo)
		{
			return std::make_tuple(std::vector<float>(), std::vector<float>(), std::vector<float>());
		}

		PrepareInfo(TInfo, stream, d_T);
		PrepareInfo(PInfo, stream, d_P);
		PrepareInfo(RHInfo, stream, d_RH);

		ThetaEKernel<<<gridSize, blockSize, 0, stream>>>(d_T, d_RH, d_P, d_prevT, d_prevRH, d_prevP, d_maxThetaE,
		                                                 d_Tresult, d_TDresult, d_Presult, d_found, N);

		size_t foundCount = thrust::count(thrust::cuda::par.on(stream), dt_found, dt_found + N, 1);

		CUDA_CHECK(cudaMemcpyAsync(d_prevT, d_T, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_prevP, d_P, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_prevRH, d_RH, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));

		CUDA_CHECK(cudaStreamSynchronize(stream));

		curLevel.Value(curLevel.Value() - 1);

		if (foundCount == N)
		{
			break;
		}
	}

	std::vector<float> Tthetae(myTargetInfo->Data().Size());
	std::vector<float> TDthetae(myTargetInfo->Data().Size());
	std::vector<float> Pthetae(myTargetInfo->Data().Size());

	CUDA_CHECK(cudaMemcpyAsync(&Tthetae[0], d_Tresult, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(&TDthetae[0], d_TDresult, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(&Pthetae[0], d_Presult, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaFree(d_T));
	CUDA_CHECK(cudaFree(d_P));
	CUDA_CHECK(cudaFree(d_RH));
	CUDA_CHECK(cudaFree(d_prevT));
	CUDA_CHECK(cudaFree(d_prevP));
	CUDA_CHECK(cudaFree(d_prevRH));
	CUDA_CHECK(cudaFree(d_maxThetaE));
	CUDA_CHECK(cudaFree(d_Tresult));
	CUDA_CHECK(cudaFree(d_TDresult));
	CUDA_CHECK(cudaFree(d_Presult));
	CUDA_CHECK(cudaFree(d_found));

	CUDA_CHECK(cudaStreamDestroy(stream));

	return std::make_tuple(Tthetae, TDthetae, Pthetae);
}

cape_source cape_cuda::Get500mMixingRatioValuesGPU(std::shared_ptr<const plugin_configuration> conf,
                                                   std::shared_ptr<info> myTargetInfo)
{
	myTargetInfo->FirstValidGrid();
	const size_t N = myTargetInfo->Data().Size();

	const int blockSize = 256;
	const int gridSize = N / blockSize + (N % blockSize == 0 ? 0 : 1);

	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	level curLevel = itsBottomLevel;

	auto h = GET_PLUGIN(hitool);

	h->Configuration(conf);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());

	modifier_mean tp, mr;

	tp.HeightInMeters(false);
	mr.HeightInMeters(false);

	info_t PInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);

	if (!PInfo || PInfo->Data().MissingCount() == PInfo->SizeLocations())
	{
		return std::make_tuple(std::vector<float>(), std::vector<float>(), std::vector<float>());
	}

	auto dPVec = VEC(PInfo);

	auto P500m = h->VerticalValue(param("P-HPA"), 500.);

	h->HeightUnit(kHPa);

	tp.LowerHeight(dPVec);
	mr.LowerHeight(dPVec);

	tp.UpperHeight(P500m);
	mr.UpperHeight(P500m);

	auto PVec = Convert(dPVec);

	float* d_Tpot = 0;
	float* d_MR = 0;
	float* d_T = 0;
	float* d_RH = 0;
	float* d_P = 0;
	float* d_TD = 0;

	CUDA_CHECK(cudaMalloc((float**)&d_Tpot, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_MR, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_T, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_RH, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_P, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_TD, N * sizeof(float)));

	InitializeArray<float>(d_Tpot, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_MR, himan::MissingFloat(), N, stream);

	std::vector<float> Tpot(N, himan::MissingFloat());
	std::vector<float> MR(N, himan::MissingFloat());

	CUDA_CHECK(cudaHostRegister(reinterpret_cast<void*>(Tpot.data()), sizeof(float) * N, 0));
	CUDA_CHECK(cudaHostRegister(reinterpret_cast<void*>(MR.data()), sizeof(float) * N, 0));
	CUDA_CHECK(cudaHostRegister(reinterpret_cast<void*>(PVec.data()), sizeof(float) * N, 0));

	CUDA_CHECK(cudaMemcpyAsync(d_P, PVec.data(), sizeof(float) * N, cudaMemcpyHostToDevice, stream));

	while (true)
	{
		auto TVec = Convert(h->VerticalValue(param("T-K"), Convert(PVec)));
		CUDA_CHECK(cudaMemcpyAsync(d_T, &TVec[0], sizeof(float) * N, cudaMemcpyHostToDevice, stream));

		auto RHVec = Convert(h->VerticalValue(param("RH-PRCNT"), Convert(PVec)));
		CUDA_CHECK(cudaMemcpyAsync(d_RH, &RHVec[0], sizeof(float) * N, cudaMemcpyHostToDevice, stream));

		MixingRatioKernel<<<gridSize, blockSize, 0, stream>>>(d_T, d_P, d_RH, d_Tpot, d_MR, N);

		CUDA_CHECK(cudaMemcpyAsync(Tpot.data(), d_Tpot, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(MR.data(), d_MR, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));

		CUDA_CHECK(cudaStreamSynchronize(stream));

		tp.Process(Convert(Tpot), Convert(PVec));
		mr.Process(Convert(MR), Convert(PVec));

		size_t foundCount = tp.HeightsCrossed();

		ASSERT(tp.HeightsCrossed() == mr.HeightsCrossed());

		if (foundCount == N)
		{
			break;
		}

		for (auto& v : PVec)
		{
			v -= 2.0;
		}
	}

	CUDA_CHECK(cudaHostUnregister(Tpot.data()));
	CUDA_CHECK(cudaHostUnregister(MR.data()));
	CUDA_CHECK(cudaHostUnregister(PVec.data()));

	CUDA_CHECK(cudaStreamSynchronize(stream));

	// Calculate averages

	Tpot = Convert(tp.Result());
	MR = Convert(mr.Result());

	// Copy averages to GPU for final calculation
	CUDA_CHECK(cudaMemcpyAsync(d_Tpot, Tpot.data(), sizeof(float) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_MR, MR.data(), sizeof(float) * N, cudaMemcpyHostToDevice, stream));

	float* d_Psurf = 0;
	CUDA_CHECK(cudaMalloc((float**)&d_Psurf, N * sizeof(float)));

	auto Psurf = Fetch(conf, myTargetInfo->Time(), itsBottomLevel, param("P-HPA"), myTargetInfo->ForecastType());
	PrepareInfo(Psurf, stream, d_Psurf);

	InitializeArray<float>(d_T, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_TD, himan::MissingFloat(), N, stream);

	std::vector<float> T(Tpot.size(), himan::MissingFloat());
	std::vector<float> TD(T.size(), himan::MissingFloat());

	MixingRatioFinalizeKernel<<<gridSize, blockSize, 0, stream>>>(d_T, d_TD, d_Psurf, d_Tpot, d_MR, N);

	CUDA_CHECK(cudaMemcpyAsync(T.data(), d_T, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(TD.data(), d_TD, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaFree(d_Tpot));
	CUDA_CHECK(cudaFree(d_MR));
	CUDA_CHECK(cudaFree(d_RH));
	CUDA_CHECK(cudaFree(d_P));
	CUDA_CHECK(cudaFree(d_T));
	CUDA_CHECK(cudaFree(d_TD));
	CUDA_CHECK(cudaFree(d_Psurf));

	CUDA_CHECK(cudaStreamDestroy(stream));

	return std::make_tuple(T, TD, Convert(VEC(Psurf)));
}

std::pair<std::vector<float>, std::vector<float>> cape_cuda::GetLFCGPU(
    const std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<info> myTargetInfo, std::vector<float>& T,
    std::vector<float>& P, std::vector<float>& TenvLCL)
{
	auto h = GET_PLUGIN(hitool);
	h->Configuration(conf);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());
	h->HeightUnit(kHPa);

	const size_t N = myTargetInfo->Data().Size();
	const int blockSize = 256;
	const int gridSize = N / blockSize + (N % blockSize == 0 ? 0 : 1);

	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	float* d_TenvLCL = 0;
	float* d_Titer = 0;
	float* d_Piter = 0;
	float* d_LCLP = 0;
	float* d_LCLT = 0;
	float* d_LFCT = 0;
	float* d_LFCP = 0;
	float* d_Tparcel = 0;
	float* d_prevTparcel = 0;
	float* d_Tenv = 0;
	float* d_Penv = 0;
	float* d_prevTenv = 0;
	float* d_prevPenv = 0;

	unsigned char* d_found = 0;

	CUDA_CHECK(cudaMalloc((float**)&d_TenvLCL, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_Piter, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_Titer, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_LCLT, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_LCLP, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_LFCT, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_LFCP, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_found, sizeof(unsigned char) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_Tparcel, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_prevTparcel, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_Tenv, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_Penv, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_prevTenv, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_prevPenv, sizeof(float) * N));

	CUDA_CHECK(cudaMemcpyAsync(d_TenvLCL, &TenvLCL[0], sizeof(float) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_Titer, &T[0], sizeof(float) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_Piter, &P[0], sizeof(float) * N, cudaMemcpyHostToDevice, stream));

	CUDA_CHECK(cudaMemcpyAsync(d_LCLT, d_Titer, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_LCLP, d_Piter, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));

	InitializeArray<float>(d_LFCT, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_LFCP, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_prevTparcel, himan::MissingFloat(), N, stream);
	InitializeArray<unsigned char>(d_found, 0, N, stream);

	// For each grid point find the hybrid level that's below LCL and then pick the lowest level
	// among all grid points; most commonly it's the lowest hybrid level

	auto levels = h->LevelForHeight(myTargetInfo->Producer(), ::Max(P));

	level curLevel = levels.first;

	auto prevPenvInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType());
	auto prevTenvInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType());

	PrepareInfo(prevTenvInfo, stream, d_prevTenv);
	PrepareInfo(prevPenvInfo, stream, d_prevPenv);

	if (cape_cuda::itsUseVirtualTemperature)
	{
		VirtualTemperatureKernel<<<gridSize, blockSize, 0, stream>>>(d_prevTenv, d_prevPenv, N);
	}

	curLevel.Value(curLevel.Value() - 1);

	std::vector<unsigned char> found(N, 0);
	std::vector<float> LFCT(N, himan::MissingFloat());
	std::vector<float> LFCP(N, himan::MissingFloat());

	thrust::device_ptr<unsigned char> dt_found = thrust::device_pointer_cast(d_found);

	for (size_t i = 0; i < N; i++)
	{
		if ((T[i] - TenvLCL[i]) > 0.001)
		{
			found[i] = 1;
			LFCT[i] = T[i];
			LFCP[i] = P[i];
		}
	}

	CUDA_CHECK(cudaMemcpyAsync(d_found, &found[0], sizeof(unsigned char) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_LFCT, &LFCT[0], sizeof(float) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_LFCP, &LFCP[0], sizeof(float) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));

	auto hPa450 = h->LevelForHeight(myTargetInfo->Producer(), 450.);
	auto stopLevel = h->LevelForHeight(myTargetInfo->Producer(), 150.);

	while (curLevel.Value() > stopLevel.first.Value())
	{
		// Get environment temperature and pressure values for this level
		auto TenvInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType());
		auto PenvInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType());

		PrepareInfo(PenvInfo, stream, d_Penv);
		PrepareInfo(TenvInfo, stream, d_Tenv);

		if (cape_cuda::itsUseVirtualTemperature)
		{
			VirtualTemperatureKernel<<<gridSize, blockSize, 0, stream>>>(d_Tenv, d_Penv, N);
		}

		// Lift the particle from previous level to this level. In the first revolution
		// of this loop the starting level is LCL. If target level level is below current level
		// (ie. we would be lowering the particle) missing value is returned.

		MoistLiftKernel<<<gridSize, blockSize, 0, stream>>>(d_Titer, d_Piter, d_Penv, d_Tparcel, N);

		LFCKernel<<<gridSize, blockSize, 0, stream>>>(d_Tenv, d_Penv, d_prevTenv, d_prevPenv, d_Tparcel, d_prevTparcel,
		                                              d_LCLT, d_LCLP, d_LFCT, d_LFCP, d_found, curLevel.Value(),
		                                              hPa450.first.Value(), N);

		CUDA_CHECK(cudaStreamSynchronize(stream));

		size_t foundCount = thrust::count(thrust::cuda::par.on(stream), dt_found, dt_found + N, 1);

		CUDA_CHECK(cudaStreamSynchronize(stream));

		if (N == foundCount)
		{
			break;
		}

		CUDA_CHECK(cudaMemcpyAsync(d_prevTparcel, d_Tparcel, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_prevTenv, d_Tenv, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_prevPenv, d_Penv, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));

		curLevel.Value(curLevel.Value() - 1);
	}

	CUDA_CHECK(cudaMemcpyAsync(LFCT.data(), d_LFCT, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(LFCP.data(), d_LFCP, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaFree(d_LCLT));
	CUDA_CHECK(cudaFree(d_LCLP));
	CUDA_CHECK(cudaFree(d_Tparcel));
	CUDA_CHECK(cudaFree(d_prevTparcel));
	CUDA_CHECK(cudaFree(d_found));
	CUDA_CHECK(cudaFree(d_Titer));
	CUDA_CHECK(cudaFree(d_Piter));
	CUDA_CHECK(cudaFree(d_TenvLCL));
	CUDA_CHECK(cudaFree(d_Penv));
	CUDA_CHECK(cudaFree(d_Tenv));
	CUDA_CHECK(cudaFree(d_prevPenv));
	CUDA_CHECK(cudaFree(d_prevTenv));

	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaFree(d_LFCT));
	CUDA_CHECK(cudaFree(d_LFCP));

	CUDA_CHECK(cudaStreamDestroy(stream));

	return std::make_pair(LFCT, LFCP);
}

void cape_cuda::GetCINGPU(const std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<info> myTargetInfo,
                          const std::vector<float>& Tsource, const std::vector<float>& Psource,
                          const std::vector<float>& TLCL, const std::vector<float>& PLCL,
                          const std::vector<float>& ZLCL, const std::vector<float>& PLFC,
                          const std::vector<float>& ZLFC)
{
	const params PParams({param("PGR-PA"), param("P-PA")});

	forecast_time ftime = myTargetInfo->Time();
	forecast_type ftype = myTargetInfo->ForecastType();

	/*
	 * Modus operandi:
	 *
	 * 1. Integrate from ground to LCL dry adiabatically
	 *
	 * This can be done always since LCL is known at all grid points
	 * (that have source data values defined).
	 *
	 * 2. Integrate from LCL to LFC moist adiabatically
	 *
	 * Note! For some points integration will fail (no LFC found)
	 *
	 * We stop integrating at first time CAPE area is found!
	 */

	level curLevel = itsBottomLevel;

	auto prevZenvInfo = Fetch(conf, ftime, curLevel, param("HL-M"), ftype);
	auto prevTenvInfo = Fetch(conf, ftime, curLevel, param("T-K"), ftype);
	auto prevPenvInfo = Fetch(conf, ftime, curLevel, param("P-HPA"), ftype);

	const size_t N = myTargetInfo->Data().Size();
	const int blockSize = 256;
	const int gridSize = N / blockSize + (N % blockSize == 0 ? 0 : 1);

	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	float* d_Psource = 0;
	float* d_Tparcel = 0;
	float* d_prevTparcel = 0;
	float* d_Piter = 0;
	float* d_prevPiter = 0;
	float* d_Titer = 0;
	float* d_prevTiter = 0;
	float* d_PLCL = 0;
	float* d_PLFC = 0;
	float* d_cinh = 0;
	float* d_prevZenv = 0;
	float* d_prevTenv = 0;
	float* d_prevPenv = 0;
	float* d_Zenv = 0;
	float* d_Tenv = 0;
	float* d_Penv = 0;

	unsigned char* d_found = 0;

	CUDA_CHECK(cudaMalloc((float**)&d_Psource, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_Tparcel, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_prevTparcel, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_Piter, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_Titer, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_PLCL, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_PLFC, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_cinh, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_prevZenv, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_prevTenv, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_prevPenv, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_Zenv, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_Tenv, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_Penv, N * sizeof(float)));

	CUDA_CHECK(cudaMalloc((unsigned char**)&d_found, N * sizeof(unsigned char)));

	PrepareInfo(prevZenvInfo, stream, d_prevZenv);
	PrepareInfo(prevTenvInfo, stream, d_prevTenv);
	PrepareInfo(prevPenvInfo, stream, d_prevPenv);

	InitializeArray<float>(d_cinh, 0., N, stream);
	InitializeArray<float>(d_Tparcel, himan::MissingFloat(), N, stream);

	CUDA_CHECK(cudaMemcpyAsync(d_prevTparcel, Tsource.data(), sizeof(float) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_Psource, Psource.data(), sizeof(float) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_Titer, Tsource.data(), sizeof(float) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_Piter, Psource.data(), sizeof(float) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_PLCL, PLCL.data(), sizeof(float) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_PLFC, PLFC.data(), sizeof(float) * N, cudaMemcpyHostToDevice, stream));

	std::vector<unsigned char> found(N, 0);

	for (size_t i = 0; i < PLFC.size(); i++)
	{
		if (IsMissing(PLFC[i]))
		{
			found[i] = true;
		}
	}

	CUDA_CHECK(cudaMemcpyAsync(d_found, &found[0], sizeof(unsigned char) * N, cudaMemcpyHostToDevice, stream));

	curLevel.Value(curLevel.Value() - 1);

	auto h = GET_PLUGIN(hitool);
	h->Configuration(conf);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());
	h->HeightUnit(kHPa);

	auto hPa100 = h->LevelForHeight(myTargetInfo->Producer(), 100.);
	thrust::device_ptr<unsigned char> dt_found = thrust::device_pointer_cast(d_found);

	while (curLevel.Value() > hPa100.first.Value())
	{
		auto ZenvInfo = Fetch(conf, ftime, curLevel, param("HL-M"), ftype);
		auto TenvInfo = Fetch(conf, ftime, curLevel, param("T-K"), ftype);
		auto PenvInfo = Fetch(conf, ftime, curLevel, param("P-HPA"), ftype);

		PrepareInfo(ZenvInfo, stream, d_Zenv);
		PrepareInfo(PenvInfo, stream, d_Penv);
		PrepareInfo(TenvInfo, stream, d_Tenv);

		LiftLCLKernel<<<gridSize, blockSize, 0, stream>>>(d_Piter, d_Titer, d_PLCL, d_Penv, d_Tparcel, N);

		CINKernel<<<gridSize, blockSize, 0, stream>>>(d_Tenv, d_prevTenv, d_Penv, d_prevPenv, d_Zenv, d_prevZenv,
		                                              d_Tparcel, d_prevTparcel, d_PLCL, d_PLFC, d_Psource, d_cinh,
		                                              d_found, cape_cuda::itsUseVirtualTemperature, N);

		size_t foundCount = thrust::count(thrust::cuda::par.on(stream), dt_found, dt_found + N, 1);
		CUDA_CHECK(cudaMemcpyAsync(d_prevTparcel, d_Tparcel, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));

		CUDA_CHECK(cudaStreamSynchronize(stream));

		if (N == foundCount)
		{
			break;
		}

		CUDA_CHECK(cudaMemcpyAsync(d_prevZenv, d_Zenv, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_prevTenv, d_Tenv, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_prevPenv, d_Penv, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));

		// preserve starting position for those grid points that have value

		CopyLFCIteratorValuesKernel<<<gridSize, blockSize, 0, stream>>>(d_Titer, d_Tparcel, d_Piter, d_Penv, N);

		curLevel.Value(curLevel.Value() - 1);
	}

	std::vector<float> cinh(N, 0);

	CUDA_CHECK(cudaMemcpyAsync(&cinh[0], d_cinh, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaFree(d_Psource));
	CUDA_CHECK(cudaFree(d_Tparcel));
	CUDA_CHECK(cudaFree(d_prevTparcel));
	CUDA_CHECK(cudaFree(d_Piter));
	CUDA_CHECK(cudaFree(d_prevPiter));
	CUDA_CHECK(cudaFree(d_Titer));
	CUDA_CHECK(cudaFree(d_prevTiter));
	CUDA_CHECK(cudaFree(d_PLCL));
	CUDA_CHECK(cudaFree(d_PLFC));
	CUDA_CHECK(cudaFree(d_prevZenv));
	CUDA_CHECK(cudaFree(d_prevPenv));
	CUDA_CHECK(cudaFree(d_prevTenv));
	CUDA_CHECK(cudaFree(d_Zenv));
	CUDA_CHECK(cudaFree(d_Penv));
	CUDA_CHECK(cudaFree(d_Tenv));
	CUDA_CHECK(cudaFree(d_found));

	CUDA_CHECK(cudaStreamSynchronize(stream));
	CUDA_CHECK(cudaFree(d_cinh));

	CUDA_CHECK(cudaStreamDestroy(stream));

	myTargetInfo->Param(CINParam);
	myTargetInfo->Data().Set(Convert(cinh));
}

void cape_cuda::GetCAPEGPU(const std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<info> myTargetInfo,
                           const std::vector<float>& T, const std::vector<float>& P)
{
	ASSERT(T.size() == P.size());

	auto h = GET_PLUGIN(hitool);

	h->Configuration(conf);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());
	h->HeightUnit(kHPa);

	// Found count determines if we have calculated all three CAPE variation for a single grid point
	std::vector<unsigned char> found(T.size(), 0);

	// No LFC --> No CAPE

	for (size_t i = 0; i < P.size(); i++)
	{
		if (IsMissing(P[i]))
		{
			found[i] = 1;
		}
	}

	const size_t N = myTargetInfo->Data().Size();
	const int blockSize = 256;
	const int gridSize = N / blockSize + (N % blockSize == 0 ? 0 : 1);

	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	float* d_CAPE = 0;
	float* d_CAPE1040 = 0;
	float* d_CAPE3km = 0;
	float* d_ELT = 0;
	float* d_ELP = 0;
	float* d_ELZ = 0;
	float* d_LastELT = 0;
	float* d_LastELP = 0;
	float* d_LastELZ = 0;
	float* d_Titer = 0;
	float* d_Piter = 0;
	float* d_prevTparcel = 0;
	float* d_Tparcel = 0;
	float* d_LFCT = 0;
	float* d_LFCP = 0;
	float* d_prevZenv = 0;
	float* d_prevPenv = 0;
	float* d_prevTenv = 0;
	float* d_Zenv = 0;
	float* d_Penv = 0;
	float* d_Tenv = 0;

	unsigned char* d_found = 0;

	CUDA_CHECK(cudaMalloc((float**)&d_CAPE, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_CAPE1040, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_CAPE3km, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_ELP, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_ELT, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_ELZ, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_LastELP, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_LastELT, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_LastELZ, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_Piter, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_Titer, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_Tparcel, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_prevTparcel, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_prevZenv, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_prevTenv, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_prevPenv, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_Zenv, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_Tenv, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_Penv, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_LFCT, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_LFCP, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_found, sizeof(unsigned char) * N));

	CUDA_CHECK(cudaMemcpyAsync(d_Titer, T.data(), sizeof(float) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_prevTparcel, d_Titer, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_Piter, P.data(), sizeof(float) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_LFCT, T.data(), sizeof(float) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_LFCP, P.data(), sizeof(float) * N, cudaMemcpyHostToDevice, stream));

	CUDA_CHECK(cudaMemcpyAsync(d_found, found.data(), sizeof(unsigned char) * N, cudaMemcpyHostToDevice, stream));

	InitializeArray<float>(d_CAPE, 0., N, stream);
	InitializeArray<float>(d_CAPE1040, 0., N, stream);
	InitializeArray<float>(d_CAPE3km, 0., N, stream);

	InitializeArray<float>(d_ELP, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_ELT, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_ELZ, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_LastELP, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_LastELT, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_LastELZ, himan::MissingFloat(), N, stream);

	// For each grid point find the hybrid level that's below LFC and then pick the lowest level
	// among all grid points

	auto levels = h->LevelForHeight(myTargetInfo->Producer(), ::Max(P));

	level curLevel = levels.first;

	auto prevZenvInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("HL-M"), myTargetInfo->ForecastType());
	auto prevTenvInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType());
	auto prevPenvInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType());

	PrepareInfo(prevZenvInfo, stream, d_prevZenv);
	PrepareInfo(prevPenvInfo, stream, d_prevPenv);
	PrepareInfo(prevTenvInfo, stream, d_prevTenv);

	if (cape_cuda::itsUseVirtualTemperature)
	{
		VirtualTemperatureKernel<<<gridSize, blockSize, 0, stream>>>(d_prevTenv, d_prevPenv, N);
	}

	curLevel.Value(curLevel.Value());

	auto stopLevel = h->LevelForHeight(myTargetInfo->Producer(), 50.);
	auto hPa450 = h->LevelForHeight(myTargetInfo->Producer(), 450.);

	thrust::device_ptr<unsigned char> dt_found = thrust::device_pointer_cast(d_found);

	info_t PenvInfo, TenvInfo, ZenvInfo;

	while (curLevel.Value() > stopLevel.first.Value())
	{
		PenvInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType());
		TenvInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType());
		ZenvInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("HL-M"), myTargetInfo->ForecastType());

		if (!PenvInfo || !TenvInfo || !ZenvInfo)
		{
			break;
		}

		PrepareInfo(ZenvInfo, stream, d_Zenv);
		PrepareInfo(PenvInfo, stream, d_Penv);
		PrepareInfo(TenvInfo, stream, d_Tenv);

		if (cape_cuda::itsUseVirtualTemperature)
		{
			VirtualTemperatureKernel<<<gridSize, blockSize, 0, stream>>>(d_Tenv, d_Penv, N);
		}

		MoistLiftKernel<<<gridSize, blockSize, 0, stream>>>(d_Titer, d_Piter, d_Penv, d_Tparcel, N);

		CAPEKernel<<<gridSize, blockSize, 0, stream>>>(d_Tenv, d_Penv, d_Zenv, d_prevTenv, d_prevPenv, d_prevZenv,
		                                               d_Tparcel, d_prevTparcel, d_LFCT, d_LFCP, d_CAPE, d_CAPE1040,
		                                               d_CAPE3km, d_ELT, d_ELP, d_ELZ, d_LastELT, d_LastELP, d_LastELZ,
		                                               d_found, curLevel.Value(), hPa450.first.Value(), N);

		size_t foundCount = thrust::count(thrust::cuda::par.on(stream), dt_found, dt_found + N, 1);

		CUDA_CHECK(cudaStreamSynchronize(stream));

		if (foundCount == N)
		{
			break;
		}

		CUDA_CHECK(cudaMemcpyAsync(d_prevTparcel, d_Tparcel, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_prevZenv, d_Zenv, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_prevTenv, d_Tenv, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_prevPenv, d_Penv, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));

		curLevel.Value(curLevel.Value() - 1);
	}

	CUDA_CHECK(cudaFree(d_Tparcel));
	CUDA_CHECK(cudaFree(d_prevTparcel));
	CUDA_CHECK(cudaFree(d_LFCT));
	CUDA_CHECK(cudaFree(d_LFCP));
	CUDA_CHECK(cudaFree(d_Piter));
	CUDA_CHECK(cudaFree(d_Titer));
	CUDA_CHECK(cudaFree(d_found));
	CUDA_CHECK(cudaFree(d_prevTenv));
	CUDA_CHECK(cudaFree(d_prevPenv));
	CUDA_CHECK(cudaFree(d_prevZenv));

	CapELValuesKernel<<<gridSize, blockSize, 0, stream>>>(d_CAPE, d_ELT, d_ELP, d_ELZ, d_LastELT, d_LastELP, d_LastELZ,
	                                                      d_Tenv, d_Penv, d_Zenv, N);

	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaFree(d_Tenv));
	CUDA_CHECK(cudaFree(d_Penv));
	CUDA_CHECK(cudaFree(d_Zenv));

	std::vector<float> CAPE(T.size());
	std::vector<float> CAPE1040(T.size());
	std::vector<float> CAPE3km(T.size());
	std::vector<float> ELT(T.size());
	std::vector<float> ELP(T.size());
	std::vector<float> ELZ(T.size());
	std::vector<float> LastELT(T.size());
	std::vector<float> LastELP(T.size());
	std::vector<float> LastELZ(T.size());

	CUDA_CHECK(cudaMemcpyAsync(CAPE.data(), d_CAPE, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(CAPE1040.data(), d_CAPE1040, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(CAPE3km.data(), d_CAPE3km, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(ELT.data(), d_ELT, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(ELP.data(), d_ELP, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(ELZ.data(), d_ELZ, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(LastELT.data(), d_LastELT, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(LastELP.data(), d_LastELP, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(LastELZ.data(), d_LastELZ, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaFree(d_CAPE));
	CUDA_CHECK(cudaFree(d_CAPE1040));
	CUDA_CHECK(cudaFree(d_CAPE3km));
	CUDA_CHECK(cudaFree(d_ELT));
	CUDA_CHECK(cudaFree(d_ELP));
	CUDA_CHECK(cudaFree(d_ELZ));
	CUDA_CHECK(cudaFree(d_LastELT));
	CUDA_CHECK(cudaFree(d_LastELP));
	CUDA_CHECK(cudaFree(d_LastELZ));

	myTargetInfo->Param(ELTParam);
	myTargetInfo->Data().Set(Convert(ELT));

	myTargetInfo->Param(ELPParam);
	myTargetInfo->Data().Set(Convert(ELP));

	myTargetInfo->Param(ELZParam);
	myTargetInfo->Data().Set(Convert(ELZ));

	myTargetInfo->Param(LastELTParam);
	myTargetInfo->Data().Set(Convert(LastELT));

	myTargetInfo->Param(LastELPParam);
	myTargetInfo->Data().Set(Convert(LastELP));

	myTargetInfo->Param(LastELZParam);
	myTargetInfo->Data().Set(Convert(LastELZ));

	myTargetInfo->Param(CAPEParam);
	myTargetInfo->Data().Set(Convert(CAPE));

	myTargetInfo->Param(CAPE1040Param);
	myTargetInfo->Data().Set(Convert(CAPE1040));

	myTargetInfo->Param(CAPE3kmParam);
	myTargetInfo->Data().Set(Convert(CAPE3km));
}
