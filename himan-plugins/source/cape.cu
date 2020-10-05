// System includes
#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/execution_policy.h>

#include "plugin_factory.h"

#include "cape.cuh"
#include "cuda_helper.h"
#include "lift.h"
#include "util.h"

#include "cuda_plugin_helper.h"
#include "forecast_time.h"
#include "level.h"
#include "timer.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "hitool.h"

#include "debug.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan;
using namespace himan::numerical_functions;
using namespace himan::plugin;

himan::level cape_cuda::itsBottomLevel;
bool cape_cuda::itsUseVirtualTemperature;

typedef std::vector<std::vector<float>> vec2d;

extern float Max(const std::vector<float>& vec);
extern std::tuple<vec2d, vec2d, vec2d> GetSampledSourceData(std::shared_ptr<const plugin_configuration> conf,
                                                            std::shared_ptr<info<float>> myTargetInfo,
                                                            const std::vector<float>& P500m,
                                                            const std::vector<float>& Psurface, const level& startLevel,
                                                            const level& stopLevel);

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

__global__ void LastLFCCopyKernel(const float* __restrict__ d_LFCT, const float* __restrict__ d_LFCP,
                                  float* __restrict__ d_LastLFCT, float* __restrict__ d_LastLFCP, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		if (IsMissing(d_LastLFCT[idx]) || d_LastLFCP[idx] > d_LFCP[idx])
		{
			d_LastLFCT[idx] = d_LFCT[idx];
			d_LastLFCP[idx] = d_LFCP[idx];
		}
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

			ASSERT(d_cinh[idx] <= 0.f);
		}
	}
}

__global__ void LFCKernel(const float* __restrict__ d_T, const float* __restrict__ d_P,
                          const float* __restrict__ d_prevT, const float* __restrict__ d_prevP,
                          float* __restrict__ d_Tparcel, const float* __restrict__ d_prevTparcel,
                          const float* __restrict__ d_LCLT, const float* __restrict__ d_LCLP,
                          float* __restrict__ d_LFCT, float* __restrict__ d_LFCP, float* __restrict__ d_LastLFCT,
                          float* __restrict__ d_LastLFCP, unsigned char* __restrict__ d_found, int d_curLevel,
                          int d_breakLevel, size_t N)
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

		if ((d_curLevel < d_breakLevel && (Tenv - Tparcel) > 30.) || (IsValid(d_LFCT[idx]) && Penv < 650.))
		{
			// Temperature gap between environment and parcel too large --> abort search.
			// Only for values higher in the atmosphere, to avoid the effects of inversion

			d_found[idx] = 1;
		}

		const float diff = Tparcel - Tenv;
		const float prevdiff = prevTparcel - prevTenv;
		const bool isFirstLFC = (diff >= 0 || fabs(diff) < 1e-4) && IsMissing(prevdiff) && IsMissing(d_LFCT[idx]);
		const bool isLastLFC = (diff >= 0 || fabs(diff) < 1e-4) && prevdiff < 0;

		if (d_found[idx] == 0 && Penv < LCLP && (isFirstLFC || isLastLFC))
		{
			if (IsMissing(prevTparcel))
			{
				prevTparcel = d_LCLT[idx];  // previous is LCL
				ASSERT(!IsMissing(d_LCLT[idx]));
			}

			float& Tresult = (IsMissing(d_LFCT[idx])) ? d_LFCT[idx] : d_LastLFCT[idx];
			float& Presult = (IsMissing(d_LFCP[idx])) ? d_LFCP[idx] : d_LastLFCP[idx];

			if (diff < 0.01)
			{
				Tresult = Tparcel;
				Presult = Penv;
			}
			else if (prevTparcel - prevTenv >= 0)
			{
				Tresult = prevTparcel;
				Presult = prevPenv;
			}
			else
			{
				auto intersection = CAPE::GetPointOfIntersection(point(Tenv, Penv), point(prevTenv, prevPenv),
				                                                 point(Tparcel, Penv), point(prevTparcel, prevPenv));

				Tresult = intersection.X();
				Presult = intersection.Y();

				if (Presult > prevPenv)
				{
					// Do not allow LFC to be below previous level; if intersection fails to put it in the correct
					// "bin" (between previous and current pressure), use the only information that certain:
					// the crossing has happened at least at current pressure
					Tresult = Tparcel;
					Presult = Penv;
				}
				else if (IsMissing(Tresult))
				{
					// Intersection not found, use exact level value
					Tresult = Tparcel;
					Presult = Penv;
				}
			}

			ASSERT(Tresult > 100);
			ASSERT(Tresult < 350);
		}
	}
}

__global__ void ThetaEKernel(float* __restrict__ d_T, const float* __restrict__ d_RH, float* __restrict__ d_P,
                             const float* __restrict__ d_prevT, const float* __restrict__ d_prevRH,
                             const float* __restrict__ d_prevP, float* __restrict__ d_ThetaE, float* __restrict__ d_TD,
                             unsigned char* __restrict__ d_found, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		float ThetaE = MissingFloat(), TD = MissingFloat();

		if (d_found[idx] == 0)
		{
			float& T = d_T[idx];
			float& P = d_P[idx];
			float RH = d_RH[idx];

			if (P < mucape_search_limit)
			{
				T = interpolation::Linear<float>(mucape_search_limit, P, d_prevP[idx], T, d_prevT[idx]);
				RH = interpolation::Linear<float>(mucape_search_limit, P, d_prevP[idx], RH, d_prevRH[idx]);

				d_found[idx] = 1;  // Make sure this is the last time we access this grid point
				P = mucape_search_limit;
			}

			TD = metutil::DewPointFromRH_<float>(T, RH);
			ThetaE = metutil::smarttool::ThetaE_<float>(T, RH, P * 100);
		}

		d_ThetaE[idx] = ThetaE;
		d_TD[idx] = TD;
	}
}

__global__ void MixingRatioKernel(const float* __restrict__ d_T, const float* __restrict__ d_P,
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
		const float RH = fminf(102., E / Es * 100);

		d_TD[idx] = metutil::DewPointFromRH_<float>(T, RH);
		d_T[idx] = T;
	}
}

__global__ void Max1D(const float* __restrict__ d_v, unsigned char* __restrict__ d_maxima, unsigned char mask_len,
                      unsigned char K, size_t N)
{
	ASSERT(mask_len % 2 == 1);

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		const unsigned char half = mask_len / 2;

		// data layout is changed here wrt to the source

		// old layout:
		// |x(0)y(0)z(0)..n(0)|x(1)y(1)z(1)..n(1)|..|x(N)y(N)z(N)..n(N)|

		// new layout:
		// |x(0)x(1)x(2)..x(K)|y(1)y(2)y(2)..y(K)|..|n(1)n(1)n(3)..n(K)|

		// beginning

		for (unsigned char i = 0; i < half; i++)
		{
			float maxv = d_v[idx];
			unsigned char maxl = 0;  // first guess

			for (unsigned char j = 1; j <= half + i; j++)
			{
				if (d_v[idx + j * N] > maxv)
				{
					maxv = d_v[idx + j * N];
					maxl = j;
				}
			}
			d_maxima[i + idx * K] = maxl;
		}

		// center

		for (unsigned char i = half; i < K - half; i++)
		{
			float maxv = d_v[idx + (i - half) * N];
			unsigned char maxl = i - half;

			for (unsigned char j = i - half + 1; j <= i + half; j++)
			{
				if (d_v[idx + j * N] > maxv)
				{
					maxv = d_v[idx + j * N];
					maxl = j;
				}
			}
			d_maxima[i + idx * K] = maxl;
		}

		// end

		for (unsigned char i = K - half; i < K; i++)
		{
			float maxv = d_v[idx + (i - half) * N];
			unsigned char maxl = i - half;

			for (unsigned char j = i - half + 1; j < K; j++)
			{
				if (d_v[idx + j * N] > maxv)
				{
					maxv = d_v[idx + j * N];
					maxl = j;
				}
			}
			d_maxima[i + idx * K] = maxl;
		}
	}
}

__global__ void MaximaLocation(const float* __restrict__ d_v, const unsigned char* __restrict__ d_maxima,
                               unsigned char* __restrict__ d_idx, size_t K, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		const int maxMax = K / 4;

		int maximaN = 0;

		for (int i = 0; i < K && maximaN < (maxMax - 1); i++)
		{
			const float v = d_v[idx + i * N];  // ThetaE value at this point in the profile

			if (i == d_maxima[i + idx * K])
			{
				if (i > 0 && v == d_v[idx + (i - 1) * N])
				{
					// Duplicate maximas (two consecutive vertical levels
					// have the same thetae value and are both maximas).
					// Disregard this higher one.
				}
				else
				{
					d_idx[maximaN + 1 + idx * maxMax] = i;
					maximaN++;
				}
			}
		}

		d_idx[idx * maxMax] = maximaN;

		// bubble sort: highest value theta e should be first

		bool passed;

		do
		{
			passed = true;

			for (int i = 2; i < maximaN + 1; i++)
			{
				unsigned char& previ = d_idx[i - 1 + idx * maxMax];
				unsigned char& curi = d_idx[i + idx * maxMax];
				float prev = d_v[previ * N + idx];
				float cur = d_v[curi * N + idx];

				if (prev < cur)
				{
					unsigned char tmpi = curi;
					curi = previ;
					previ = tmpi;

					passed = false;
				}
			}
		} while (!passed);
	}
}

__global__ void MeanKernel(const float* __restrict__ d_Tpot, const float* __restrict__ d_MR,
                           const float* __restrict__ d_prevTpot, const float* __restrict__ d_prevMR,
                           float* __restrict__ d_meanTpot, float* __restrict__ d_meanMR, float* __restrict__ d_range,
                           size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		if (IsValid(d_Tpot[idx]))
		{
			// trapezoidal integration
			d_range[idx] += 1;
			d_meanTpot[idx] += (d_prevTpot[idx] + d_Tpot[idx]) * 0.5;
			d_meanMR[idx] += (d_prevMR[idx] + d_MR[idx]) * 0.5;
		}
	}
}

__global__ void MeanFinalizeKernel(float* __restrict__ d_meanTpot, float* __restrict__ d_meanMR,
                                   const float* __restrict__ d_range, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		d_meanTpot[idx] = d_meanTpot[idx] / d_range[idx];
		d_meanMR[idx] = d_meanMR[idx] / d_range[idx];
	}
}

cape_multi_source cape_cuda::GetNHighestThetaEValuesGPU(const std::shared_ptr<const plugin_configuration>& conf,
                                                        std::shared_ptr<info<float>> myTargetInfo, int n)
{
	himan::level curLevel = itsBottomLevel;

	auto h = GET_PLUGIN(hitool);

	h->Configuration(conf);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());
	h->HeightUnit(kHPa);

	// We need to get the number of layers so we can preallocate
	// a suitable sized array.

	const auto stopLevel = h->LevelForHeight(myTargetInfo->Producer(), mucape_search_limit);
	const auto levelSpan = curLevel.Value() - stopLevel.second.Value();

	const size_t N = myTargetInfo->Data().Size();
	const int blockSize = 256;
	const int gridSize = N / blockSize + (N % blockSize == 0 ? 0 : 1);

	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	float* d_T = 0;
	float* d_TD = 0;
	float* d_P = 0;
	float* d_ThetaE = 0;
	float* d_RH = 0;
	float* d_prevT = 0;
	float* d_prevP = 0;
	float* d_prevRH = 0;

	unsigned char* d_found = 0;

	CUDA_CHECK(cudaMalloc((float**)&d_T, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_TD, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_P, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_ThetaE, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_found, sizeof(unsigned char) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_RH, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_prevT, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_prevP, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_prevRH, sizeof(float) * N));

	InitializeArray<float>(d_T, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_TD, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_P, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_prevT, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_prevP, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_prevRH, himan::MissingFloat(), N, stream);

	InitializeArray<unsigned char>(d_found, 0, N, stream);

	thrust::device_ptr<unsigned char> dt_found = thrust::device_pointer_cast(d_found);

	// profiles are create as flattened vectors
	// in order the insertion to be as fast as possible, the layout is such:
	// |x(0)y(0)z(0)..n(0)|x(1)y(1)z(1)..n(1)|..|x(N)y(N)z(N)..n(N)|

	std::vector<float> ThetaEProfile(levelSpan * N), TProfile(levelSpan * N), PProfile(levelSpan * N),
	    TDProfile(levelSpan * N);

	CUDA_CHECK(cudaHostRegister(reinterpret_cast<void*>(ThetaEProfile.data()), sizeof(float) * levelSpan * N, 0));
	CUDA_CHECK(cudaHostRegister(reinterpret_cast<void*>(TProfile.data()), sizeof(float) * N * levelSpan, 0));
	CUDA_CHECK(cudaHostRegister(reinterpret_cast<void*>(TDProfile.data()), sizeof(float) * N, 0));
	CUDA_CHECK(cudaHostRegister(reinterpret_cast<void*>(PProfile.data()), sizeof(float) * N, 0));

	const param TParam("T-K");
	const param RHParam("RH-PRCNT");
	const param PParam("P-HPA");

	size_t K = 0;  // this will hold the number of levels read (should match what we calculated previously)
	while (true)
	{
		auto TInfo = cuda::Fetch<float>(conf, myTargetInfo->Time(), curLevel, TParam, myTargetInfo->ForecastType());
		auto RHInfo = cuda::Fetch<float>(conf, myTargetInfo->Time(), curLevel, RHParam, myTargetInfo->ForecastType());
		auto PInfo = cuda::Fetch<float>(conf, myTargetInfo->Time(), curLevel, PParam, myTargetInfo->ForecastType());

		if (!TInfo || !RHInfo || !PInfo)
		{
			CUDA_CHECK(cudaHostUnregister(ThetaEProfile.data()));
			CUDA_CHECK(cudaHostUnregister(TProfile.data()));
			CUDA_CHECK(cudaHostUnregister(TDProfile.data()));
			CUDA_CHECK(cudaHostUnregister(PProfile.data()));

			CUDA_CHECK(cudaFree(d_T));
			CUDA_CHECK(cudaFree(d_P));
			CUDA_CHECK(cudaFree(d_RH));
			CUDA_CHECK(cudaFree(d_prevT));
			CUDA_CHECK(cudaFree(d_prevP));
			CUDA_CHECK(cudaFree(d_prevRH));
			CUDA_CHECK(cudaFree(d_ThetaE));
			CUDA_CHECK(cudaFree(d_TD));
			CUDA_CHECK(cudaFree(d_found));

			return cape_multi_source();
		}

		cuda::PrepareInfo(TInfo, d_T, stream, conf->UseCacheForReads());
		cuda::PrepareInfo(PInfo, d_P, stream, conf->UseCacheForReads());
		cuda::PrepareInfo(RHInfo, d_RH, stream, conf->UseCacheForReads());

		ThetaEKernel<<<gridSize, blockSize, 0, stream>>>(d_T, d_RH, d_P, d_prevT, d_prevRH, d_prevP, d_ThetaE, d_TD,
		                                                 d_found, N);

		size_t foundCount = thrust::count(thrust::cuda::par.on(stream), dt_found, dt_found + N, 1);

		CUDA_CHECK(cudaMemcpyAsync(&ThetaEProfile[K * N], d_ThetaE, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(&TProfile[K * N], d_T, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(&TDProfile[K * N], d_TD, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(&PProfile[K * N], d_P, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));

		CUDA_CHECK(cudaStreamSynchronize(stream));

		curLevel.Value(curLevel.Value() - 1);
		K++;

		if (foundCount == N || levelSpan == K)
		{
			break;
		}

		CUDA_CHECK(cudaMemcpyAsync(d_prevT, d_T, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_prevP, d_P, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_prevRH, d_RH, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));
	}

	CUDA_CHECK(cudaHostUnregister(ThetaEProfile.data()));
	CUDA_CHECK(cudaHostUnregister(TProfile.data()));
	CUDA_CHECK(cudaHostUnregister(TDProfile.data()));
	CUDA_CHECK(cudaHostUnregister(PProfile.data()));

	CUDA_CHECK(cudaFree(d_T));
	CUDA_CHECK(cudaFree(d_P));
	CUDA_CHECK(cudaFree(d_RH));
	CUDA_CHECK(cudaFree(d_prevT));
	CUDA_CHECK(cudaFree(d_prevP));
	CUDA_CHECK(cudaFree(d_prevRH));
	CUDA_CHECK(cudaFree(d_ThetaE));
	CUDA_CHECK(cudaFree(d_TD));
	CUDA_CHECK(cudaFree(d_found));

	// Check comments from cape.cpp
	vec2d Tret(n), TDret(n), Pret(n);

	for (size_t j = 0; j < static_cast<size_t>(n); j++)
	{
		Tret[j].resize(N, MissingFloat());
		TDret[j].resize(N, MissingFloat());
		Pret[j].resize(N, MissingFloat());
	}

	float* d_v = 0;
	unsigned char* d_maxima = 0;
	unsigned char* d_idxs = 0;

	CUDA_CHECK(cudaMalloc((float**)&d_v, sizeof(float) * N * K));  // Actual ThetaE values
	CUDA_CHECK(cudaMalloc((unsigned char**)&d_maxima,
	                      sizeof(unsigned char) * N * K));  // Local maxima locations in the profile

	CUDA_CHECK(cudaMemcpyAsync(d_v, ThetaEProfile.data(), sizeof(float) * N * K, cudaMemcpyHostToDevice, stream));

	Max1D<<<gridSize, blockSize, 0, stream>>>(d_v, d_maxima, 5, K, N);

	// maximum number of maximas we expect to find in the profile
	const size_t maxMax = K / 4;

	CUDA_CHECK(cudaMalloc((unsigned char**)&d_idxs, sizeof(unsigned char) * N * maxMax));

	MaximaLocation<<<gridSize, blockSize, 0, stream>>>(d_v, d_maxima, d_idxs, K, N);

	std::vector<unsigned char> idxs(N * maxMax);

	CUDA_CHECK(cudaMemcpyAsync(&idxs[0], d_idxs, sizeof(unsigned char) * N * maxMax, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaFree(d_v));
	CUDA_CHECK(cudaFree(d_maxima));
	CUDA_CHECK(cudaFree(d_idxs));

	for (size_t i = 0; i < N; i++)
	{
		const size_t s = i * maxMax;     // start index of this grid point
		const size_t maximaN = idxs[s];  // number of maximas found (at most maxMax)

		ASSERT(maximaN > 0);
		ASSERT(maximaN <= maxMax);

		// Remove maximas that are too high in the atmosphere
		size_t newMaximaN = 0;
		size_t offset = 0;
		for (size_t j = 0; j < maximaN; j++)
		{
			const size_t sidx = 1 + s + j;            // index in the array where maxima index is found
			const unsigned char maxidx = idxs[sidx];  // index in the vertical profile where the maxima was found

			if (PProfile[maxidx * N + i] < mucape_maxima_search_limit)
			{
				offset++;
				continue;
			}
			newMaximaN++;
			idxs[sidx - offset] = maxidx;
		}

		for (size_t j = 0; j < min(static_cast<size_t>(n), newMaximaN); j++)
		{
			const size_t sidx = 1 + s + j;  // index in the array where maxima index is found
			const short maxidx =
			    static_cast<short>(idxs[sidx]);  // index in the vertical profile where the maxima was found

			ASSERT(static_cast<unsigned>(maxidx) <= K);

			// Copy values from max theta e levels for further processing

			Tret[j][i] = TProfile[maxidx * N + i];
			TDret[j][i] = TDProfile[maxidx * N + i];
			Pret[j][i] = PProfile[maxidx * N + i];

			ASSERT(IsValid(Tret[j][i]));
			ASSERT(IsValid(TDret[j][i]));
			ASSERT(IsValid(Pret[j][i]));
		}
	}

	CUDA_CHECK(cudaStreamDestroy(stream));

	return std::make_tuple(Tret, TDret, Pret);
}

cape_source cape_cuda::Get500mMixingRatioValuesGPU(std::shared_ptr<const plugin_configuration>& conf,
                                                   std::shared_ptr<info<float>> myTargetInfo)
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

	auto PInfo = cuda::Fetch<double>(conf, myTargetInfo->Time(), curLevel, PParam, myTargetInfo->ForecastType(), false);

	if (!PInfo || PInfo->Data().MissingCount() == PInfo->SizeLocations())
	{
		return std::make_tuple(std::vector<float>(), std::vector<float>(), std::vector<float>());
	}

	auto dPVec = VEC(PInfo);

	auto P500m = h->VerticalValue<double>(PParam, 500.);
	auto stopLevel = h->LevelForHeight(myTargetInfo->Producer(), 500.);

	h->HeightUnit(kHPa);

	tp.LowerHeight(dPVec);
	mr.LowerHeight(dPVec);

	tp.UpperHeight(P500m);
	mr.UpperHeight(P500m);

	auto PVec = util::Convert<double, float>(dPVec);

	auto sourceData = GetSampledSourceData(conf, myTargetInfo, util::Convert<double, float>(P500m), PVec,
	                                       itsBottomLevel, stopLevel.second);

	float* d_Tpot = 0;
	float* d_MR = 0;
	float* d_prevTpot = 0;
	float* d_prevMR = 0;
	float* d_meanTpot = 0;
	float* d_meanMR = 0;
	float* d_range = 0;
	float* d_T = 0;
	float* d_RH = 0;
	float* d_P = 0;
	float* d_TD = 0;

	CUDA_CHECK(cudaMalloc((float**)&d_Tpot, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_MR, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_prevTpot, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_prevMR, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_meanTpot, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_meanMR, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_range, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_T, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_RH, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_P, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc((float**)&d_TD, N * sizeof(float)));

	InitializeArray<float>(d_meanTpot, 0.f, N, stream);
	InitializeArray<float>(d_meanMR, 0.f, N, stream);
	InitializeArray<float>(d_range, 0.f, N, stream);

	const auto& Psample = std::get<0>(sourceData);
	const auto& Tsample = std::get<1>(sourceData);
	const auto& RHsample = std::get<2>(sourceData);

	std::vector<float> T(N, MissingFloat());
	std::vector<float> RH(N, MissingFloat());
	std::vector<float> P(N, MissingFloat());

	CUDA_CHECK(cudaHostRegister(reinterpret_cast<void*>(T.data()), sizeof(float) * N, 0));
	CUDA_CHECK(cudaHostRegister(reinterpret_cast<void*>(P.data()), sizeof(float) * N, 0));
	CUDA_CHECK(cudaHostRegister(reinterpret_cast<void*>(RH.data()), sizeof(float) * N, 0));

	unsigned int k = 0;

	while (true)
	{
		CUDA_CHECK(cudaStreamSynchronize(stream));

		for (size_t i = 0; i < N; i++)
		{
			if (k >= Psample[i].size())
			{
				T[i] = RH[i] = P[i] = MissingFloat();
				continue;
			}

			T[i] = Tsample[i][k];
			RH[i] = RHsample[i][k];
			P[i] = Psample[i][k];
		}

		if (static_cast<unsigned int>(count_if(P.begin(), P.end(), [](const float& v) { return IsMissing(v); })) ==
		    P.size())
		{
			break;
		}

		CUDA_CHECK(cudaMemcpyAsync(d_T, T.data(), sizeof(float) * N, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_RH, RH.data(), sizeof(float) * N, cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_P, P.data(), sizeof(float) * N, cudaMemcpyHostToDevice, stream));

		MixingRatioKernel<<<gridSize, blockSize, 0, stream>>>(d_T, d_P, d_RH, d_Tpot, d_MR, N);

		if (k == 0)
		{
			CUDA_CHECK(cudaMemcpyAsync(d_prevTpot, d_Tpot, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));
			CUDA_CHECK(cudaMemcpyAsync(d_prevMR, d_MR, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));

			k++;
			continue;
		}

		MeanKernel<<<gridSize, blockSize, 0, stream>>>(d_Tpot, d_MR, d_prevTpot, d_prevMR, d_meanTpot, d_meanMR,
		                                               d_range, N);

		CUDA_CHECK(cudaMemcpyAsync(d_prevTpot, d_Tpot, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_prevMR, d_MR, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));

		k++;
	}

	CUDA_CHECK(cudaHostUnregister(T.data()));
	CUDA_CHECK(cudaHostUnregister(P.data()));
	CUDA_CHECK(cudaHostUnregister(RH.data()));

	MeanFinalizeKernel<<<gridSize, blockSize, 0, stream>>>(d_meanTpot, d_meanMR, d_range, N);

	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaFree(d_Tpot));
	CUDA_CHECK(cudaFree(d_MR));
	CUDA_CHECK(cudaFree(d_prevTpot));
	CUDA_CHECK(cudaFree(d_prevMR));
	CUDA_CHECK(cudaFree(d_range));

	float* d_Psurf = 0;
	CUDA_CHECK(cudaMalloc((float**)&d_Psurf, N * sizeof(float)));

	auto Psurf = cuda::Fetch<float>(conf, myTargetInfo->Time(), itsBottomLevel, PParam, myTargetInfo->ForecastType());
	cuda::PrepareInfo(Psurf, d_Psurf, stream, conf->UseCacheForReads());

	std::vector<float> TD(T.size());

	MixingRatioFinalizeKernel<<<gridSize, blockSize, 0, stream>>>(d_T, d_TD, d_Psurf, d_meanTpot, d_meanMR, N);

	CUDA_CHECK(cudaMemcpyAsync(T.data(), d_T, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(TD.data(), d_TD, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaFree(d_meanTpot));
	CUDA_CHECK(cudaFree(d_meanMR));
	CUDA_CHECK(cudaFree(d_RH));
	CUDA_CHECK(cudaFree(d_P));
	CUDA_CHECK(cudaFree(d_T));
	CUDA_CHECK(cudaFree(d_TD));
	CUDA_CHECK(cudaFree(d_Psurf));

	CUDA_CHECK(cudaStreamDestroy(stream));
	return std::make_tuple(T, TD, VEC(Psurf));
}

std::vector<std::pair<std::vector<float>, std::vector<float>>> cape_cuda::GetLFCGPU(
    const std::shared_ptr<const plugin_configuration>& conf, std::shared_ptr<info<float>> myTargetInfo,
    std::vector<float>& T, std::vector<float>& P, std::vector<float>& TenvLCL)
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

	float* d_LCLP = 0;
	float* d_LCLT = 0;
	float* d_LFCT = 0;
	float* d_LFCP = 0;
	float* d_LastLFCT = 0;
	float* d_LastLFCP = 0;
	float* d_Tparcel = 0;
	float* d_prevTparcel = 0;
	float* d_Tenv = 0;
	float* d_Penv = 0;
	float* d_prevTenv = 0;
	float* d_prevPenv = 0;

	unsigned char* d_found = 0;

	CUDA_CHECK(cudaMalloc((float**)&d_LCLT, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_LCLP, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_LFCT, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_LFCP, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_LastLFCT, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_LastLFCP, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_found, sizeof(unsigned char) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_Tparcel, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_prevTparcel, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_Tenv, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_Penv, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_prevTenv, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_prevPenv, sizeof(float) * N));

	CUDA_CHECK(cudaMemcpyAsync(d_LCLT, &T[0], sizeof(float) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_LCLP, &P[0], sizeof(float) * N, cudaMemcpyHostToDevice, stream));

	InitializeArray<float>(d_LFCT, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_LFCP, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_LastLFCT, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_LastLFCP, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_prevTparcel, himan::MissingFloat(), N, stream);
	InitializeArray<unsigned char>(d_found, 0, N, stream);

	// For each grid point find the hybrid level that's below LCL and then pick the lowest level
	// among all grid points; most commonly it's the lowest hybrid level

	auto levels = h->LevelForHeight(myTargetInfo->Producer(), ::Max(P));

	level curLevel = levels.first;

	auto prevPenvInfo = cuda::Fetch<float>(conf, myTargetInfo->Time(), curLevel, PParam, myTargetInfo->ForecastType());
	auto prevTenvInfo = cuda::Fetch<float>(conf, myTargetInfo->Time(), curLevel, TParam, myTargetInfo->ForecastType());

	cuda::PrepareInfo(prevTenvInfo, d_prevTenv, stream, conf->UseCacheForReads());
	cuda::PrepareInfo(prevPenvInfo, d_prevPenv, stream, conf->UseCacheForReads());

	if (cape_cuda::itsUseVirtualTemperature)
	{
		VirtualTemperatureKernel<<<gridSize, blockSize, 0, stream>>>(d_prevTenv, d_prevPenv, N);
	}

	curLevel.Value(curLevel.Value() - 1);

	std::vector<unsigned char> found(N, 0);
	std::vector<float> LFCT(N, himan::MissingFloat());
	std::vector<float> LFCP(N, himan::MissingFloat());
	std::vector<float> LastLFCT(N);
	std::vector<float> LastLFCP(N);

	thrust::device_ptr<unsigned char> dt_found = thrust::device_pointer_cast(d_found);

	for (size_t i = 0; i < N; i++)
	{
		if ((T[i] - TenvLCL[i]) > 0.0001)
		{
			LFCT[i] = T[i];
			LFCP[i] = P[i];
		}
	}

	CUDA_CHECK(cudaMemcpyAsync(d_found, &found[0], sizeof(unsigned char) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_LFCT, &LFCT[0], sizeof(float) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_LFCP, &LFCP[0], sizeof(float) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));

	auto hPa450 = h->LevelForHeight(myTargetInfo->Producer(), 450.);
	auto stopLevel = h->LevelForHeight(myTargetInfo->Producer(), 250.);

	while (curLevel.Value() > stopLevel.first.Value())
	{
		// Get environment temperature and pressure values for this level
		auto TenvInfo = cuda::Fetch<float>(conf, myTargetInfo->Time(), curLevel, TParam, myTargetInfo->ForecastType());
		auto PenvInfo = cuda::Fetch<float>(conf, myTargetInfo->Time(), curLevel, PParam, myTargetInfo->ForecastType());

		cuda::PrepareInfo(PenvInfo, d_Penv, stream, conf->UseCacheForReads());
		cuda::PrepareInfo(TenvInfo, d_Tenv, stream, conf->UseCacheForReads());

		if (cape_cuda::itsUseVirtualTemperature)
		{
			VirtualTemperatureKernel<<<gridSize, blockSize, 0, stream>>>(d_Tenv, d_Penv, N);
		}

		// Lift the particle from previous level to this level. In the first revolution
		// of this loop the starting level is LCL. If target level level is below current level
		// (ie. we would be lowering the particle) missing value is returned.

		MoistLiftKernel<<<gridSize, blockSize, 0, stream>>>(d_LCLT, d_LCLP, d_Penv, d_Tparcel, N);

		LFCKernel<<<gridSize, blockSize, 0, stream>>>(d_Tenv, d_Penv, d_prevTenv, d_prevPenv, d_Tparcel, d_prevTparcel,
		                                              d_LCLT, d_LCLP, d_LFCT, d_LFCP, d_LastLFCT, d_LastLFCP, d_found,
		                                              curLevel.Value(), hPa450.first.Value(), N);

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

	LastLFCCopyKernel<<<gridSize, blockSize, 0, stream>>>(d_LFCT, d_LFCP, d_LastLFCT, d_LastLFCP, N);

	CUDA_CHECK(cudaMemcpyAsync(LFCT.data(), d_LFCT, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(LFCP.data(), d_LFCP, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(LastLFCT.data(), d_LastLFCT, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(LastLFCP.data(), d_LastLFCP, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaFree(d_LCLT));
	CUDA_CHECK(cudaFree(d_LCLP));
	CUDA_CHECK(cudaFree(d_Tparcel));
	CUDA_CHECK(cudaFree(d_prevTparcel));
	CUDA_CHECK(cudaFree(d_found));
	CUDA_CHECK(cudaFree(d_Penv));
	CUDA_CHECK(cudaFree(d_Tenv));
	CUDA_CHECK(cudaFree(d_prevPenv));
	CUDA_CHECK(cudaFree(d_prevTenv));

	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaFree(d_LFCT));
	CUDA_CHECK(cudaFree(d_LFCP));
	CUDA_CHECK(cudaFree(d_LastLFCT));
	CUDA_CHECK(cudaFree(d_LastLFCP));

	CUDA_CHECK(cudaStreamDestroy(stream));

	return {std::make_pair(LFCT, LFCP), std::make_pair(LastLFCT, LastLFCP)};
}

std::vector<float> cape_cuda::GetCINGPU(const std::shared_ptr<const plugin_configuration>& conf,
                                        std::shared_ptr<info<float>> myTargetInfo, const std::vector<float>& Tsource,
                                        const std::vector<float>& Psource, const std::vector<float>& PLCL,
                                        const std::vector<float>& PLFC, const std::vector<float>& ZLFC)
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

	auto prevZenvInfo = cuda::Fetch<float>(conf, ftime, curLevel, ZParam, ftype);
	auto prevTenvInfo = cuda::Fetch<float>(conf, ftime, curLevel, TParam, ftype);
	auto prevPenvInfo = cuda::Fetch<float>(conf, ftime, curLevel, PParam, ftype);

	const size_t N = myTargetInfo->Data().Size();
	const int blockSize = 256;
	const int gridSize = N / blockSize + (N % blockSize == 0 ? 0 : 1);

	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	float* d_Psource = 0;
	float* d_Tparcel = 0;
	float* d_prevTparcel = 0;
	float* d_Tsource = 0;
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
	CUDA_CHECK(cudaMalloc((float**)&d_Tsource, N * sizeof(float)));
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

	cuda::PrepareInfo(prevZenvInfo, d_prevZenv, stream, conf->UseCacheForReads());
	cuda::PrepareInfo(prevTenvInfo, d_prevTenv, stream, conf->UseCacheForReads());
	cuda::PrepareInfo(prevPenvInfo, d_prevPenv, stream, conf->UseCacheForReads());

	InitializeArray<float>(d_cinh, 0., N, stream);
	InitializeArray<float>(d_Tparcel, himan::MissingFloat(), N, stream);

	CUDA_CHECK(cudaMemcpyAsync(d_prevTparcel, Tsource.data(), sizeof(float) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_Psource, Psource.data(), sizeof(float) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_Tsource, d_prevTparcel, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));
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
		auto ZenvInfo = cuda::Fetch<float>(conf, ftime, curLevel, ZParam, ftype);
		auto TenvInfo = cuda::Fetch<float>(conf, ftime, curLevel, TParam, ftype);
		auto PenvInfo = cuda::Fetch<float>(conf, ftime, curLevel, PParam, ftype);

		cuda::PrepareInfo(ZenvInfo, d_Zenv, stream, conf->UseCacheForReads());
		cuda::PrepareInfo(PenvInfo, d_Penv, stream, conf->UseCacheForReads());
		cuda::PrepareInfo(TenvInfo, d_Tenv, stream, conf->UseCacheForReads());

		LiftLCLKernel<<<gridSize, blockSize, 0, stream>>>(d_Psource, d_Tsource, d_PLCL, d_Penv, d_Tparcel, N);

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

		curLevel.Value(curLevel.Value() - 1);
	}

	std::vector<float> cinh(N, 0);

	CUDA_CHECK(cudaMemcpyAsync(&cinh[0], d_cinh, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaFree(d_Psource));
	CUDA_CHECK(cudaFree(d_Tparcel));
	CUDA_CHECK(cudaFree(d_prevTparcel));
	CUDA_CHECK(cudaFree(d_Tsource));
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

	return cinh;
}

CAPEdata cape_cuda::GetCAPEGPU(const std::shared_ptr<const plugin_configuration>& conf,
                               std::shared_ptr<info<float>> myTargetInfo, const std::vector<float>& T,
                               const std::vector<float>& P)
{
	ASSERT(T.size() == P.size());

	auto h = GET_PLUGIN(hitool);

	h->Configuration(conf);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());
	h->HeightUnit(kHPa);

	// Typically LFC value is not found to all grid points -- in many cases as few as
	// 10% of grid points have LFC defined. Therefore we do some memory-optimization
	// here: instead of allocating and processing whole grids with 90% missing data,
	// we 'compact' the data with a bitmap.
	//
	// The LFC pressure field is read and each missing grid point is marked with a '1'.
	// The effective grid size in the processing phase is then the number of ones in the
	// bitmap field. Every time a a new field is read from database, it is processed so
	// that grid points who's bitmap value is zero are removed.
	//
	// At the end of processing the grid is again 'expanded' so that the grid point values
	// in the compact grid are moved to correct locations.

	std::vector<unsigned char> bitmap(T.size(), 1);

	// No LFC --> No CAPE

	for (size_t i = 0; i < P.size(); i++)
	{
		if (IsMissing(P[i]))
		{
			bitmap[i] = 0;
		}
	}

	const size_t N = count(bitmap.begin(), bitmap.end(), 1);
	const size_t NB = bitmap.size();

	// 'bitmap' variable is immutable; 'found' variable gets filled up with
	// ones as processing proceeds

	std::vector<unsigned char> found(N, 0);

	logger log("CAPEGPU");

	log.Info("Compacting data with a ratio of " + std::to_string(static_cast<double>(N) / NB) + " (smaller is better)");

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
	float* d_prevTparcel = 0;
	float* d_Tparcel = 0;
	float* d_LFCT = 0;
	float* d_LFCP = 0;
	float* d_origLFCT = 0;
	float* d_origLFCP = 0;
	float* d_prevZenv = 0;
	float* d_prevPenv = 0;
	float* d_prevTenv = 0;
	float* d_Zenv = 0;
	float* d_Penv = 0;
	float* d_Tenv = 0;
	float* d_origZenv = 0;
	float* d_origPenv = 0;
	float* d_origTenv = 0;

	unsigned char* d_found = 0;
	unsigned char* d_bitmap = 0;

	CUDA_CHECK(cudaMalloc((float**)&d_CAPE, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_CAPE1040, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_CAPE3km, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_ELP, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_ELT, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_ELZ, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_LastELP, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_LastELT, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_LastELZ, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_Tparcel, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_prevTparcel, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_prevZenv, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_prevTenv, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_prevPenv, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_Zenv, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_Tenv, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_Penv, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_origZenv, sizeof(float) * NB));
	CUDA_CHECK(cudaMalloc((float**)&d_origTenv, sizeof(float) * NB));
	CUDA_CHECK(cudaMalloc((float**)&d_origPenv, sizeof(float) * NB));
	CUDA_CHECK(cudaMalloc((float**)&d_LFCT, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_LFCP, sizeof(float) * N));
	CUDA_CHECK(cudaMalloc((float**)&d_origLFCT, sizeof(float) * NB));
	CUDA_CHECK(cudaMalloc((float**)&d_origLFCP, sizeof(float) * NB));
	CUDA_CHECK(cudaMalloc((unsigned char**)&d_found, sizeof(unsigned char) * N));
	CUDA_CHECK(cudaMalloc((unsigned char**)&d_bitmap, sizeof(unsigned char) * NB));

	CUDA_CHECK(cudaMemcpyAsync(d_origLFCP, P.data(), sizeof(float) * NB, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_origLFCT, T.data(), sizeof(float) * NB, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_bitmap, bitmap.data(), sizeof(unsigned char) * NB, cudaMemcpyHostToDevice, stream));

	CUDA_CHECK(cudaMemcpyAsync(d_prevTparcel, d_LFCT, sizeof(float) * N, cudaMemcpyDeviceToDevice, stream));

	CUDA_CHECK(cudaMemcpyAsync(d_bitmap, bitmap.data(), sizeof(unsigned char) * NB, cudaMemcpyHostToDevice, stream));

	auto bitmapHot = [] __device__(const unsigned char& u) { return u == 1; };

	// https://thrust.github.io/doc/group__stream__compaction.html#ga36d9d6ed8e17b442c1fd8dc40bd515d5
	thrust::copy_if(thrust::cuda::par.on(stream), d_origLFCT, d_origLFCT + NB, d_bitmap, d_LFCT, bitmapHot);
	thrust::copy_if(thrust::cuda::par.on(stream), d_origLFCP, d_origLFCP + NB, d_bitmap, d_LFCP, bitmapHot);

	InitializeArray<unsigned char>(d_found, 0., N, stream);
	InitializeArray<float>(d_CAPE, 0., N, stream);
	InitializeArray<float>(d_CAPE1040, 0., N, stream);
	InitializeArray<float>(d_CAPE3km, 0., N, stream);

	InitializeArray<float>(d_ELP, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_ELT, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_ELZ, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_LastELP, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_LastELT, himan::MissingFloat(), N, stream);
	InitializeArray<float>(d_LastELZ, himan::MissingFloat(), N, stream);

	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaFree(d_origLFCT));
	CUDA_CHECK(cudaFree(d_origLFCP));

	// For each grid point find the hybrid level that's below LFC and then pick the lowest level
	// among all grid points

	auto levels = h->LevelForHeight(myTargetInfo->Producer(), ::Max(P));

	level curLevel = levels.first;

	auto prevZenvInfo = cuda::Fetch<float>(conf, myTargetInfo->Time(), curLevel, ZParam, myTargetInfo->ForecastType());
	auto prevTenvInfo = cuda::Fetch<float>(conf, myTargetInfo->Time(), curLevel, TParam, myTargetInfo->ForecastType());
	auto prevPenvInfo = cuda::Fetch<float>(conf, myTargetInfo->Time(), curLevel, PParam, myTargetInfo->ForecastType());

	// "orig" variables are just as a temporary placeholder
	cuda::PrepareInfo(prevZenvInfo, d_origZenv, stream, conf->UseCacheForReads());
	cuda::PrepareInfo(prevPenvInfo, d_origPenv, stream, conf->UseCacheForReads());
	cuda::PrepareInfo(prevTenvInfo, d_origTenv, stream, conf->UseCacheForReads());

	thrust::copy_if(thrust::cuda::par.on(stream), d_origZenv, d_origZenv + NB, d_bitmap, d_prevZenv, bitmapHot);
	thrust::copy_if(thrust::cuda::par.on(stream), d_origPenv, d_origPenv + NB, d_bitmap, d_prevPenv, bitmapHot);
	thrust::copy_if(thrust::cuda::par.on(stream), d_origTenv, d_origTenv + NB, d_bitmap, d_prevTenv, bitmapHot);

	if (cape_cuda::itsUseVirtualTemperature)
	{
		VirtualTemperatureKernel<<<gridSize, blockSize, 0, stream>>>(d_prevTenv, d_prevPenv, N);
	}

	curLevel.Value(curLevel.Value());

	auto stopLevel = h->LevelForHeight(myTargetInfo->Producer(), 50.);
	auto hPa450 = h->LevelForHeight(myTargetInfo->Producer(), 450.);

	thrust::device_ptr<unsigned char> dt_found = thrust::device_pointer_cast(d_found);

	std::shared_ptr<info<float>> PenvInfo, TenvInfo, ZenvInfo;

	while (curLevel.Value() > stopLevel.first.Value())
	{
		PenvInfo = cuda::Fetch<float>(conf, myTargetInfo->Time(), curLevel, PParam, myTargetInfo->ForecastType());
		TenvInfo = cuda::Fetch<float>(conf, myTargetInfo->Time(), curLevel, TParam, myTargetInfo->ForecastType());
		ZenvInfo = cuda::Fetch<float>(conf, myTargetInfo->Time(), curLevel, ZParam, myTargetInfo->ForecastType());

		if (!PenvInfo || !TenvInfo || !ZenvInfo)
		{
			break;
		}

		cuda::PrepareInfo<float>(ZenvInfo, d_origZenv, stream, conf->UseCacheForReads());
		cuda::PrepareInfo<float>(PenvInfo, d_origPenv, stream, conf->UseCacheForReads());
		cuda::PrepareInfo<float>(TenvInfo, d_origTenv, stream, conf->UseCacheForReads());

		thrust::copy_if(thrust::cuda::par.on(stream), d_origZenv, d_origZenv + NB, d_bitmap, d_Zenv, bitmapHot);
		thrust::copy_if(thrust::cuda::par.on(stream), d_origPenv, d_origPenv + NB, d_bitmap, d_Penv, bitmapHot);
		thrust::copy_if(thrust::cuda::par.on(stream), d_origTenv, d_origTenv + NB, d_bitmap, d_Tenv, bitmapHot);

		if (cape_cuda::itsUseVirtualTemperature)
		{
			VirtualTemperatureKernel<<<gridSize, blockSize, 0, stream>>>(d_Tenv, d_Penv, N);
		}

		MoistLiftKernel<<<gridSize, blockSize, 0, stream>>>(d_LFCT, d_LFCP, d_Penv, d_Tparcel, N);

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
	CUDA_CHECK(cudaFree(d_found));
	CUDA_CHECK(cudaFree(d_bitmap));
	CUDA_CHECK(cudaFree(d_prevTenv));
	CUDA_CHECK(cudaFree(d_prevPenv));
	CUDA_CHECK(cudaFree(d_prevZenv));
	CUDA_CHECK(cudaFree(d_origTenv));
	CUDA_CHECK(cudaFree(d_origPenv));
	CUDA_CHECK(cudaFree(d_origZenv));

	CapELValuesKernel<<<gridSize, blockSize, 0, stream>>>(d_CAPE, d_ELT, d_ELP, d_ELZ, d_LastELT, d_LastELP, d_LastELZ,
	                                                      d_Tenv, d_Penv, d_Zenv, N);

	CUDA_CHECK(cudaStreamSynchronize(stream));
	CUDA_CHECK(cudaFree(d_Tenv));
	CUDA_CHECK(cudaFree(d_Penv));
	CUDA_CHECK(cudaFree(d_Zenv));

	std::vector<float> CAPE(NB, 0);
	std::vector<float> CAPE1040(NB, 0);
	std::vector<float> CAPE3km(NB, 0);
	std::vector<float> ELT(NB, himan::MissingFloat());
	std::vector<float> ELP(NB, himan::MissingFloat());
	std::vector<float> ELZ(NB, himan::MissingFloat());
	std::vector<float> LastELT(NB, himan::MissingFloat());
	std::vector<float> LastELP(NB, himan::MissingFloat());
	std::vector<float> LastELZ(NB, himan::MissingFloat());

	// intermediate containers where data is copied before expansion
	std::vector<float> bm_CAPE(N);
	std::vector<float> bm_CAPE1040(N);
	std::vector<float> bm_CAPE3km(N);
	std::vector<float> bm_ELT(N);
	std::vector<float> bm_ELP(N);
	std::vector<float> bm_ELZ(N);
	std::vector<float> bm_LastELT(N);
	std::vector<float> bm_LastELP(N);
	std::vector<float> bm_LastELZ(N);

	CUDA_CHECK(cudaMemcpyAsync(bm_CAPE.data(), d_CAPE, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(bm_CAPE1040.data(), d_CAPE1040, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(bm_CAPE3km.data(), d_CAPE3km, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(bm_ELT.data(), d_ELT, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(bm_ELP.data(), d_ELP, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(bm_ELZ.data(), d_ELZ, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(bm_LastELT.data(), d_LastELT, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(bm_LastELP.data(), d_LastELP, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(bm_LastELZ.data(), d_LastELZ, sizeof(float) * N, cudaMemcpyDeviceToHost, stream));

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

	// "expand" data again with bitmap

	for (size_t i = 0, j = 0; i < NB; i++)
	{
		if (bitmap[i])
		{
			CAPE[i] = bm_CAPE[j];
			CAPE1040[i] = bm_CAPE1040[j];
			CAPE3km[i] = bm_CAPE3km[j];
			ELT[i] = bm_ELT[j];
			ELP[i] = bm_ELP[j];
			ELZ[i] = bm_ELZ[j];
			LastELT[i] = bm_LastELT[j];
			LastELP[i] = bm_LastELP[j];
			LastELZ[i] = bm_LastELZ[j];

			j++;
		}
	}

	return make_tuple(ELT, ELP, ELZ, LastELT, LastELP, LastELZ, CAPE, CAPE1040, CAPE3km);
}
