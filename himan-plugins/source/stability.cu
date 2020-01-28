#include "cuda_plugin_helper.h"
#include "lift.h"
#include "plugin_factory.h"
#include "stability.cuh"
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/execution_policy.h>

#define HIMAN_AUXILIARY_INCLUDE
#include "hitool.h"
#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan;

himan::level himan::plugin::stability_cuda::itsBottomLevel;

namespace STABILITY
{
extern std::vector<double> Shear(std::shared_ptr<himan::plugin::hitool>& h, const himan::param& par, double lowerHeight,
                                 double upperHeight, size_t N);
std::vector<double> Shear(std::shared_ptr<himan::plugin::hitool>& h, const himan::param& par,
                          const std::vector<double>& lowerHeight, const std::vector<double>& upperHeight);
std::pair<std::vector<double>, std::vector<double>> GetEBSLevelData(std::shared_ptr<const plugin_configuration>& conf,
                                                                    info_t& myTargetInfo,
                                                                    std::shared_ptr<plugin::hitool>& h,
                                                                    const level& sourceLevel, const level& targetLevel);
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

using himan::MissingDouble;
static size_t memsize;
static int gridSize;
static int blockSize;

__global__ void ThetaEKernel(cdarr_t d_tstart, cdarr_t d_rhstart, cdarr_t d_pstart, cdarr_t d_tstop, cdarr_t d_rhstop,
                             cdarr_t d_pstop, darr_t d_thetaediff, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		const double tstart = d_tstart[idx];
		const double rhstart = d_rhstart[idx];
		const double pstart = d_pstart[idx];

		const double tstop = d_tstop[idx];
		const double rhstop = d_rhstop[idx];
		const double pstop = d_pstop[idx];

		const double thetaestart = himan::metutil::smarttool::ThetaE_(tstart, rhstart, pstart * 100);
		const double thetaestop = himan::metutil::smarttool::ThetaE_(tstop, rhstop, pstop * 100);

		d_thetaediff[idx] = thetaestart - thetaestop;
	}
}

__global__ void LiftedIndicesKernel(cdarr_t d_t850, cdarr_t d_t500, cdarr_t d_t500m, cdarr_t d_td850, cdarr_t d_td500m,
                                    cdarr_t d_p500m, darr_t d_si, darr_t d_li, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		const double t_li = himan::metutil::Lift_(d_p500m[idx], d_t500m[idx], d_td500m[idx], 50000.);
		const double t_si = himan::metutil::Lift_(85000., d_t850[idx], d_td850[idx], 50000.);

		d_si[idx] = d_t500[idx] - t_si;
		d_li[idx] = d_t500[idx] - t_li;
	}
}

__global__ void BulkShearKernel(cdarr_t d_u, cdarr_t d_v, darr_t d_bs, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		const double u = d_u[idx];
		const double v = d_v[idx];

		d_bs[idx] = hypot(u, v);
	}
}

__global__ void CAPEShearKernel(cdarr_t d_cape, cdarr_t d_ebs, darr_t d_capes, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		const double cape = d_cape[idx];
		const double ebs = d_ebs[idx];

		d_capes[idx] = ebs * __dsqrt_rn(cape);
	}
}

__global__ void RHToTDKernel(cdarr_t d_t, darr_t d_rh, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		d_rh[idx] = himan::metutil::DewPointFromRH_(d_t[idx], d_rh[idx]);
	}
}

__global__ void StormRelativeHelicityKernel(darr_t d_srh, cdarr_t d_u, cdarr_t d_v, cdarr_t d_pu, cdarr_t d_pv,
                                            cdarr_t d_uid, cdarr_t d_vid, cdarr_t d_z, cdarr_t d_pz,
                                            unsigned char* __restrict__ d_found, double stopHeight, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N && d_found[idx] == 0)
	{
		const double z = d_z[idx];
		const double pz = d_pz[idx];
		const double uid = d_uid[idx];
		const double vid = d_vid[idx];
		const double pu = d_pu[idx];
		const double pv = d_pv[idx];
		double u = d_u[idx];
		double v = d_v[idx];

		if (z > stopHeight)
		{
			u = himan::numerical_functions::interpolation::Linear<double>(stopHeight, pz, z, pu, u);
			v = himan::numerical_functions::interpolation::Linear<double>(stopHeight, pz, z, pv, v);

			d_found[idx] = 1;
		}

		const double res = ((uid - pu) * (pv - v)) - ((vid - pv) * (pu - u));

		if (!himan::IsMissingDouble(d_srh[idx]))
		{
			d_srh[idx] -= res;
		}
	}
}

__global__ void UVIdVectorKernel(darr_t d_uid, darr_t d_vid, cdarr_t d_uavg, cdarr_t d_vavg, cdarr_t d_ushr,
                                 cdarr_t d_vshr, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		STABILITY::UVId(d_ushr[idx], d_vshr[idx], d_uavg[idx], d_vavg[idx], d_uid[idx], d_vid[idx]);
	}
}

__global__ void EHIKernel(cdarr_t d_cape, cdarr_t d_srh01, darr_t d_ehi, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		d_ehi[idx] = d_cape[idx] * d_srh01[idx] / 160000.;
	}
}

__global__ void BRNKernel(cdarr_t d_cape, cdarr_t d_u6, cdarr_t d_v6, cdarr_t d_u05, cdarr_t d_v05, darr_t d_brn,
                          size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		d_brn[idx] = STABILITY::BRN(d_cape[idx], d_u6[idx], d_v6[idx], d_u05[idx], d_v05[idx]);
	}
}

__global__ void CSIKernel(cdarr_t d_mucape, cdarr_t d_mlcape, cdarr_t d_mulpl, cdarr_t d_muebs, cdarr_t d_mlebs,
                          darr_t d_csi, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		double cape = MissingDouble();
		double ebs = MissingDouble();

		if (d_mulpl[idx] >= 250. && d_mucape[idx] > 10.)
		{
			cape = d_mucape[idx];
			ebs = d_muebs[idx];
		}
		else if (d_mulpl[idx] < 250. && d_mlcape[idx] > 10.)
		{
			cape = d_mlcape[idx];
			ebs = d_mlebs[idx];
		}

		d_csi[idx] = (ebs * sqrt(2 * cape)) * 0.1;

		if (ebs <= 15.)
		{
			d_csi[idx] += 0.025 * cape * (-0.06666 * ebs + 1);
		}
	}
}

void CalculateBulkShear(std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<info<double>> myTargetInfo,
                        std::shared_ptr<himan::plugin::hitool> h, cudaStream_t& stream)
{
	using namespace himan;

	double* d_bs = 0;
	double* d_capes = 0;
	double* d_u = 0;
	double* d_v = 0;

	CUDA_CHECK(cudaMalloc((void**)&d_bs, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_capes, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_u, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_v, memsize));

	const size_t N = myTargetInfo->SizeLocations();

	try
	{
		auto u = STABILITY::Shear(h, UParam, 10, 1000, N);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_u, (const void*)u.data(), memsize, cudaMemcpyHostToDevice, stream));

		auto v = STABILITY::Shear(h, VParam, 10, 1000, N);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_v, (const void*)v.data(), memsize, cudaMemcpyHostToDevice, stream));

		BulkShearKernel<<<gridSize, blockSize, 0, stream>>>(d_u, d_v, d_bs, N);

		myTargetInfo->Find<param>(BSParam);
		myTargetInfo->Find<level>(OneKMLevel);
		cuda::ReleaseInfo(myTargetInfo, d_bs, stream);

		u = STABILITY::Shear(h, UParam, 10, 3000, N);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_u, (const void*)u.data(), memsize, cudaMemcpyHostToDevice, stream));

		v = STABILITY::Shear(h, VParam, 10, 3000, N);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_v, (const void*)v.data(), memsize, cudaMemcpyHostToDevice, stream));

		BulkShearKernel<<<gridSize, blockSize, 0, stream>>>(d_u, d_v, d_bs, N);

		myTargetInfo->Find<level>(ThreeKMLevel);
		cuda::ReleaseInfo(myTargetInfo, d_bs, stream);

		u = STABILITY::Shear(h, UParam, 10, 6000, N);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_u, (const void*)u.data(), memsize, cudaMemcpyHostToDevice, stream));

		v = STABILITY::Shear(h, VParam, 10, 6000, N);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_v, (const void*)v.data(), memsize, cudaMemcpyHostToDevice, stream));

		BulkShearKernel<<<gridSize, blockSize, 0, stream>>>(d_u, d_v, d_bs, N);

		myTargetInfo->Find<level>(SixKMLevel);
		cuda::ReleaseInfo(myTargetInfo, d_bs, stream);

		// Maximum effective bulk shear

		const auto muMaxEBSLevels = STABILITY::GetEBSLevelData(conf, myTargetInfo, h, MaxThetaELevel, MaxWindLevel);

		u = STABILITY::Shear(h, UParam, muMaxEBSLevels.first, muMaxEBSLevels.second);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_u, (const void*)u.data(), memsize, cudaMemcpyHostToDevice, stream));

		v = STABILITY::Shear(h, VParam, muMaxEBSLevels.first, muMaxEBSLevels.second);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_v, (const void*)v.data(), memsize, cudaMemcpyHostToDevice, stream));

		BulkShearKernel<<<gridSize, blockSize, 0, stream>>>(d_u, d_v, d_bs, N);

		myTargetInfo->Find<level>(MaxWindLevel);
		myTargetInfo->Find<param>(EBSParam);
		cuda::ReleaseInfo(myTargetInfo, d_bs, stream);

		// CAPE shear

		const auto muEBSLevels = STABILITY::GetEBSLevelData(conf, myTargetInfo, h, MaxThetaELevel, Height0Level);

		u = STABILITY::Shear(h, UParam, muEBSLevels.first, muEBSLevels.second);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_u, (const void*)u.data(), memsize, cudaMemcpyHostToDevice, stream));

		v = STABILITY::Shear(h, VParam, muEBSLevels.first, muEBSLevels.second);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_v, (const void*)v.data(), memsize, cudaMemcpyHostToDevice, stream));

		BulkShearKernel<<<gridSize, blockSize, 0, stream>>>(d_u, d_v, d_bs, N);

		auto CAPEInfo = cuda::Fetch<double>(conf, myTargetInfo->Time(), MaxThetaELevel, param("CAPE-JKG"),
		                                    myTargetInfo->ForecastType(), false);

		if (!CAPEInfo)
		{
			throw himan::kFileDataNotFound;
		}

		CUDA_CHECK(cudaStreamSynchronize(stream));
		cuda::PrepareInfo(CAPEInfo, d_u, stream, conf->UseCacheForReads());

		CAPEShearKernel<<<gridSize, blockSize, 0, stream>>>(d_u, d_bs, d_capes, N);

		myTargetInfo->Find<param>(CAPESParam);
		myTargetInfo->Find<level>(Height0Level);
		cuda::ReleaseInfo(myTargetInfo, d_capes, stream);
	}
	catch (HPExceptionType& e)
	{
	}

	if (d_bs)
		CUDA_CHECK(cudaFree(d_bs));
	if (d_capes)
		CUDA_CHECK(cudaFree(d_capes));
	if (d_u)
		CUDA_CHECK(cudaFree(d_u));
	if (d_v)
		CUDA_CHECK(cudaFree(d_v));
}

void StormRelativeHelicity(std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<info<double>> myTargetInfo,
                           std::shared_ptr<himan::plugin::hitool> h, double* d_srh, double stopHeight,
                           cudaStream_t& stream)
{
	using namespace himan;
	using himan::plugin::stability_cuda::itsBottomLevel;

	const size_t N = myTargetInfo->SizeLocations();

	InitializeArray<double>(d_srh, 0, N, stream);

	double* d_uid = 0;
	double* d_vid = 0;
	double* d_uavg = 0;
	double* d_vavg = 0;
	double* d_ushr = 0;
	double* d_vshr = 0;
	double* d_pu = 0;
	double* d_pv = 0;
	double* d_pt = 0;
	double* d_pz = 0;
	double* d_u = 0;
	double* d_v = 0;
	double* d_z = 0;
	unsigned char* d_found = 0;

	// First fetch U and V identity vectors, the same are used for both 1km and 3km
	// helicity.

	try
	{
		CUDA_CHECK(cudaMalloc((void**)&d_uid, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_vid, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_uavg, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_vavg, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_ushr, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_vshr, memsize));

		// average wind
		auto Uavg = h->VerticalAverage<double>(UParam, 10, 6000);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_uavg, (const void*)Uavg.data(), memsize, cudaMemcpyHostToDevice, stream));

		auto Vavg = h->VerticalAverage<double>(VParam, 10, 6000);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_vavg, (const void*)Vavg.data(), memsize, cudaMemcpyHostToDevice, stream));

		// shear
		auto Ushear = STABILITY::Shear(h, UParam, 10, 6000, Uavg.size());
		CUDA_CHECK(cudaMemcpyAsync((void*)d_ushr, (const void*)Ushear.data(), memsize, cudaMemcpyHostToDevice, stream));

		auto Vshear = STABILITY::Shear(h, VParam, 10, 6000, Uavg.size());
		CUDA_CHECK(cudaMemcpyAsync((void*)d_vshr, (const void*)Vshear.data(), memsize, cudaMemcpyHostToDevice, stream));

		UVIdVectorKernel<<<gridSize, blockSize, 0, stream>>>(d_uid, d_vid, d_uavg, d_vavg, d_ushr, d_vshr, N);

		CUDA_CHECK(cudaStreamSynchronize(stream));

		// U&V id vector source data is not needed anymore
		CUDA_CHECK(cudaFree(d_uavg));
		CUDA_CHECK(cudaFree(d_vavg));
		CUDA_CHECK(cudaFree(d_ushr));
		CUDA_CHECK(cudaFree(d_vshr));
	}
	catch (const himan::HPExceptionType& e)
	{
		if (e == himan::kFileDataNotFound)
		{
			CUDA_CHECK(cudaFree(d_uid));
			CUDA_CHECK(cudaFree(d_vid));
			CUDA_CHECK(cudaFree(d_uavg));
			CUDA_CHECK(cudaFree(d_vavg));
			CUDA_CHECK(cudaFree(d_ushr));
			CUDA_CHECK(cudaFree(d_vshr));
			return;
		}
	}

	try
	{
		CUDA_CHECK(cudaMalloc((void**)&d_pu, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_pv, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_pt, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_pz, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_u, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_v, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_z, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_found, N * sizeof(unsigned char)));

		InitializeArray<unsigned char>(d_found, 0, N, stream);

		auto prevUInfo =
		    cuda::Fetch<double>(conf, myTargetInfo->Time(), itsBottomLevel, UParam, myTargetInfo->ForecastType());
		auto prevVInfo =
		    cuda::Fetch<double>(conf, myTargetInfo->Time(), itsBottomLevel, VParam, myTargetInfo->ForecastType());
		auto prevZInfo =
		    cuda::Fetch<double>(conf, myTargetInfo->Time(), itsBottomLevel, HLParam, myTargetInfo->ForecastType());

		if (!prevUInfo || !prevVInfo || !prevZInfo)
		{
			throw himan::kFileDataNotFound;
		}

		cuda::PrepareInfo(prevUInfo, d_pu, stream, conf->UseCacheForReads());
		cuda::PrepareInfo(prevVInfo, d_pv, stream, conf->UseCacheForReads());
		cuda::PrepareInfo(prevZInfo, d_pz, stream, conf->UseCacheForReads());

		thrust::device_ptr<unsigned char> dt_found = thrust::device_pointer_cast(d_found);

		level curLevel = itsBottomLevel;

		while (curLevel.Value() > 0)
		{
			curLevel.Value(curLevel.Value() - 1);

			auto UInfo =
			    cuda::Fetch<double>(conf, myTargetInfo->Time(), curLevel, UParam, myTargetInfo->ForecastType());
			auto VInfo =
			    cuda::Fetch<double>(conf, myTargetInfo->Time(), curLevel, VParam, myTargetInfo->ForecastType());
			auto ZInfo =
			    cuda::Fetch<double>(conf, myTargetInfo->Time(), curLevel, HLParam, myTargetInfo->ForecastType());

			if (!UInfo || !VInfo || !ZInfo)
			{
				break;
			}

			cuda::PrepareInfo(UInfo, d_u, stream, conf->UseCacheForReads());
			cuda::PrepareInfo(VInfo, d_v, stream, conf->UseCacheForReads());
			cuda::PrepareInfo(ZInfo, d_z, stream, conf->UseCacheForReads());

			StormRelativeHelicityKernel<<<gridSize, blockSize, 0, stream>>>(d_srh, d_u, d_v, d_pu, d_pv, d_uid, d_vid,
			                                                                d_z, d_pz, d_found, stopHeight, N);

			size_t foundCount = thrust::count(thrust::cuda::par.on(stream), dt_found, dt_found + N, 1);
			CUDA_CHECK(cudaStreamSynchronize(stream));

			if (foundCount == N)
			{
				break;
			}

			CUDA_CHECK(cudaMemcpyAsync(d_pu, d_u, memsize, cudaMemcpyDeviceToDevice, stream));
			CUDA_CHECK(cudaMemcpyAsync(d_pv, d_v, memsize, cudaMemcpyDeviceToDevice, stream));
			CUDA_CHECK(cudaMemcpyAsync(d_pz, d_z, memsize, cudaMemcpyDeviceToDevice, stream));
		}

		CUDA_CHECK(cudaFree(d_uid));
		CUDA_CHECK(cudaFree(d_vid));
		CUDA_CHECK(cudaFree(d_pu));
		CUDA_CHECK(cudaFree(d_pv));
		CUDA_CHECK(cudaFree(d_pt));
		CUDA_CHECK(cudaFree(d_pz));
		CUDA_CHECK(cudaFree(d_u));
		CUDA_CHECK(cudaFree(d_v));
		CUDA_CHECK(cudaFree(d_z));
	}
	catch (const himan::HPExceptionType& e)
	{
		if (e == himan::kFileDataNotFound)
		{
			CUDA_CHECK(cudaFree(d_uid));
			CUDA_CHECK(cudaFree(d_vid));
			CUDA_CHECK(cudaFree(d_pu));
			CUDA_CHECK(cudaFree(d_pv));
			CUDA_CHECK(cudaFree(d_pt));
			CUDA_CHECK(cudaFree(d_pz));
			CUDA_CHECK(cudaFree(d_u));
			CUDA_CHECK(cudaFree(d_v));
			CUDA_CHECK(cudaFree(d_z));
		}
		else
		{
			throw;
		}
	}

	thrust::device_ptr<double> dt_srh = thrust::device_pointer_cast(d_srh);
	thrust::replace(dt_srh, dt_srh + N, 0., himan::MissingDouble());
}

void EnergyHelicityIndex(std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<info<double>> myTargetInfo,
                         double* d_srh, double* d_ehi, cudaStream_t& stream)
{
	double* d_cape = 0;

	try
	{
		auto CAPEInfo = cuda::Fetch<double>(conf, myTargetInfo->Time(), himan::level(himan::kHeightLayer, 500, 0),
		                                    himan::param("CAPE-JKG"), myTargetInfo->ForecastType());

		if (!CAPEInfo)
		{
			return;
		}

		CUDA_CHECK(cudaMalloc((void**)&d_cape, memsize));

		cuda::PrepareInfo(CAPEInfo, d_cape, stream, conf->UseCacheForReads());

		EHIKernel<<<gridSize, blockSize, 0, stream>>>(d_cape, d_srh, d_ehi, myTargetInfo->SizeLocations());

		CUDA_CHECK(cudaFree(d_cape));
	}
	catch (const himan::HPExceptionType& e)
	{
		if (e == himan::kFileDataNotFound)
		{
		}
	}
}

void CalculateHelicity(std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<info<double>> myTargetInfo,
                       std::shared_ptr<himan::plugin::hitool> h, cudaStream_t& stream)
{
	double* d_srh = 0;
	double* d_ehi = 0;

	CUDA_CHECK(cudaMalloc((void**)&d_srh, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_ehi, memsize));

	myTargetInfo->Find<param>(SRHParam);
	myTargetInfo->Find<level>(ThreeKMLevel);

	StormRelativeHelicity(conf, myTargetInfo, h, d_srh, 3000, stream);
	cuda::ReleaseInfo(myTargetInfo, d_srh, stream);

	myTargetInfo->Find<level>(OneKMLevel);

	StormRelativeHelicity(conf, myTargetInfo, h, d_srh, 1000, stream);
	cuda::ReleaseInfo(myTargetInfo, d_srh, stream);

	myTargetInfo->Find<param>(EHIParam);

	EnergyHelicityIndex(conf, myTargetInfo, d_srh, d_ehi, stream);
	cuda::ReleaseInfo(myTargetInfo, d_ehi, stream);

	CUDA_CHECK(cudaFree(d_srh));
	CUDA_CHECK(cudaFree(d_ehi));
}

void CalculateBulkRichardsonNumber(std::shared_ptr<const plugin_configuration> conf,
                                   std::shared_ptr<info<double>> myTargetInfo, std::shared_ptr<himan::plugin::hitool> h,
                                   cudaStream_t& stream)
{
	double* d_brn = 0;
	double* d_cape = 0;
	double* d_u6 = 0;
	double* d_v6 = 0;
	double* d_u05 = 0;
	double* d_v05 = 0;

	try
	{
		auto CAPEInfo = cuda::Fetch<double>(conf, myTargetInfo->Time(), himan::level(himan::kHeightLayer, 500, 0),
		                                    himan::param("CAPE-JKG"), myTargetInfo->ForecastType());

		if (!CAPEInfo)
		{
			return;
		}

		CUDA_CHECK(cudaMalloc((void**)&d_cape, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_u6, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_v6, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_u05, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_v05, memsize));

		cuda::PrepareInfo(CAPEInfo, d_cape, stream, conf->UseCacheForReads());

		auto U6 = h->VerticalAverage<double>(UParam, 10, 6000);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_u6, (const void*)U6.data(), memsize, cudaMemcpyHostToDevice, stream));

		auto V6 = h->VerticalAverage<double>(VParam, 10, 6000);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_v6, (const void*)V6.data(), memsize, cudaMemcpyHostToDevice, stream));

		auto U05 = h->VerticalAverage<double>(UParam, 10, 500);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_u05, (const void*)U05.data(), memsize, cudaMemcpyHostToDevice, stream));

		auto V05 = h->VerticalAverage<double>(VParam, 10, 500);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_v05, (const void*)V05.data(), memsize, cudaMemcpyHostToDevice, stream));

		CUDA_CHECK(cudaMalloc((void**)&d_brn, memsize));

		BRNKernel<<<gridSize, blockSize, 0, stream>>>(d_cape, d_u6, d_v6, d_u05, d_v05, d_brn,
		                                              myTargetInfo->SizeLocations());

		myTargetInfo->Find<param>(BRNParam);
		myTargetInfo->Find<level>(SixKMLevel);

		cuda::ReleaseInfo(myTargetInfo, d_brn, stream);

		CUDA_CHECK(cudaFree(d_brn));
		CUDA_CHECK(cudaFree(d_cape));
		CUDA_CHECK(cudaFree(d_u6));
		CUDA_CHECK(cudaFree(d_v6));
		CUDA_CHECK(cudaFree(d_u05));
		CUDA_CHECK(cudaFree(d_v05));
	}
	catch (const himan::HPExceptionType& e)
	{
		if (e == himan::kFileDataNotFound)
		{
			if (d_cape)
				CUDA_CHECK(cudaFree(d_cape));
			if (d_u6)
				CUDA_CHECK(cudaFree(d_u6));
			if (d_v6)
				CUDA_CHECK(cudaFree(d_v6));
			if (d_u05)
				CUDA_CHECK(cudaFree(d_u05));
			if (d_v05)
				CUDA_CHECK(cudaFree(d_v05));
		}
	}
}

void CalculateLiftedIndices(std::shared_ptr<const plugin_configuration> conf,
                            std::shared_ptr<info<double>> myTargetInfo, std::shared_ptr<himan::plugin::hitool> h,
                            cudaStream_t& stream)
{
	double* d_si = 0;
	double* d_li = 0;
	double* d_t500m = 0;
	double* d_td500m = 0;
	double* d_p500m = 0;
	double* d_t500 = 0;
	double* d_t850 = 0;
	double* d_td850 = 0;

	auto T850Info = cuda::Fetch<double>(conf, myTargetInfo->Time(), P850Level, TParam, myTargetInfo->ForecastType());
	auto T500Info = cuda::Fetch<double>(conf, myTargetInfo->Time(), P500Level, TParam, myTargetInfo->ForecastType());
	auto TD850Info = cuda::Fetch<double>(conf, myTargetInfo->Time(), P850Level, TDParam, myTargetInfo->ForecastType());

	if (!T850Info || !T500Info || !TD850Info)
	{
		return;
	}

	CUDA_CHECK(cudaMalloc((void**)&d_t500, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_t850, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_td850, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_si, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_li, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_t500m, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_td500m, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_p500m, memsize));

	cuda::PrepareInfo(T850Info, d_t850, stream, conf->UseCacheForReads());
	cuda::PrepareInfo(T500Info, d_t500, stream, conf->UseCacheForReads());
	cuda::PrepareInfo(TD850Info, d_td850, stream, conf->UseCacheForReads());

	const size_t N = myTargetInfo->SizeLocations();

	// CUDA_CHECK(cudaStreamSynchronize(stream));

	auto T500m = h->VerticalAverage<double>(TParam, 0, 500.);
	CUDA_CHECK(cudaMemcpyAsync((void*)d_t500m, (const void*)T500m.data(), memsize, cudaMemcpyHostToDevice, stream));

	auto P500m = h->VerticalAverage<double>(PParam, 0., 500.);
	CUDA_CHECK(cudaMemcpyAsync((void*)d_p500m, (const void*)P500m.data(), memsize, cudaMemcpyHostToDevice, stream));

	std::vector<double> TD500m;

	try
	{
		TD500m = h->VerticalAverage<double>(himan::param("TD-K"), 0, 500.);
		CUDA_CHECK(
		    cudaMemcpyAsync((void*)d_td500m, (const void*)TD500m.data(), memsize, cudaMemcpyHostToDevice, stream));
	}
	catch (const himan::HPExceptionType& e)
	{
		if (e == himan::kFileDataNotFound)
		{
			try
			{
				TD500m = h->VerticalAverage<double>(RHParam, 0, 500.);
				CUDA_CHECK(cudaMemcpyAsync((void*)d_td500m, (const void*)TD500m.data(), memsize, cudaMemcpyHostToDevice,
				                           stream));

				RHToTDKernel<<<gridSize, blockSize, 0, stream>>>(d_t500m, d_td500m, TD500m.size());
			}
			catch (const himan::HPExceptionType& e)
			{
				if (e == himan::kFileDataNotFound)
				{
					return;
				}
			}
		}
	}

	if (P500m[0] < 1500)
	{
		MultiplyWith<double>(d_p500m, 100, N, stream);
	}

	LiftedIndicesKernel<<<gridSize, blockSize, 0, stream>>>(d_t850, d_t500, d_t500m, d_td850, d_td500m, d_p500m, d_si,
	                                                        d_li, N);

	myTargetInfo->Find<level>(Height0Level);
	myTargetInfo->Find<param>(LIParam);
	cuda::ReleaseInfo(myTargetInfo, d_li, stream);

	myTargetInfo->Find<param>(SIParam);
	cuda::ReleaseInfo(myTargetInfo, d_si, stream);

	CUDA_CHECK(cudaFree(d_t850));
	CUDA_CHECK(cudaFree(d_t500));
	CUDA_CHECK(cudaFree(d_td850));
	CUDA_CHECK(cudaFree(d_li));
	CUDA_CHECK(cudaFree(d_si));
	CUDA_CHECK(cudaFree(d_t500m));
	CUDA_CHECK(cudaFree(d_td500m));
	CUDA_CHECK(cudaFree(d_p500m));
}

void CalculateThetaEIndices(std::shared_ptr<info<double>> myTargetInfo, std::shared_ptr<himan::plugin::hitool> h,
                            cudaStream_t& stream)
{
	double* d_tstart = 0;
	double* d_rhstart = 0;
	double* d_pstart = 0;
	double* d_tstop = 0;
	double* d_rhstop = 0;
	double* d_pstop = 0;

	double* d_thetaediff = 0;

	try
	{
		CUDA_CHECK(cudaMalloc((void**)&d_tstart, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_rhstart, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_pstart, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_tstop, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_rhstop, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_pstop, memsize));

		CUDA_CHECK(cudaMalloc((void**)&d_thetaediff, memsize));

		auto T3000 = h->VerticalValue<double>(TParam, 3000.);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_tstop, (const void*)T3000.data(), memsize, cudaMemcpyHostToDevice, stream));

		auto RH3000 = h->VerticalValue<double>(RHParam, 3000.);
		CUDA_CHECK(
		    cudaMemcpyAsync((void*)d_rhstop, (const void*)RH3000.data(), memsize, cudaMemcpyHostToDevice, stream));

		auto P3000 = h->VerticalValue<double>(PParam, 3000.);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_pstop, (const void*)P3000.data(), memsize, cudaMemcpyHostToDevice, stream));

		auto T2 = h->VerticalValue<double>(TParam, 2.);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_tstart, (const void*)T2.data(), memsize, cudaMemcpyHostToDevice, stream));

		auto RH2 = h->VerticalValue<double>(RHParam, 2.);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_rhstart, (const void*)RH2.data(), memsize, cudaMemcpyHostToDevice, stream));

		auto P2 = h->VerticalValue<double>(PParam, 2.);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_pstart, (const void*)P2.data(), memsize, cudaMemcpyHostToDevice, stream));

		ThetaEKernel<<<gridSize, blockSize, 0, stream>>>(d_tstart, d_rhstart, d_pstart, d_tstop, d_rhstop, d_pstop,
		                                                 d_thetaediff, myTargetInfo->SizeLocations());

		myTargetInfo->Find<level>(ThreeKMLevel);
		myTargetInfo->Find<param>(TPEParam);
		cuda::ReleaseInfo(myTargetInfo, d_thetaediff, stream);

		CUDA_CHECK(cudaFree(d_tstop));
		CUDA_CHECK(cudaFree(d_rhstop));
		CUDA_CHECK(cudaFree(d_pstop));
		CUDA_CHECK(cudaFree(d_tstart));
		CUDA_CHECK(cudaFree(d_rhstart));
		CUDA_CHECK(cudaFree(d_pstart));

		CUDA_CHECK(cudaFree(d_thetaediff));
	}
	catch (const himan::HPExceptionType& e)
	{
		if (e == himan::kFileDataNotFound)
		{
			return;
		}
	}
}

void CalculateConvectiveSeverityIndex(std::shared_ptr<const plugin_configuration> conf,
                                      std::shared_ptr<info<double>> myTargetInfo,
                                      std::shared_ptr<himan::plugin::hitool> h, cudaStream_t& stream)
{
	double* d_mucape = 0;
	double* d_mlcape = 0;
	double* d_mulpl = 0;
	double* d_muebs = 0;
	double* d_mlebs = 0;

	double* d_csi = 0;

	try
	{
		auto muCAPEInfo = cuda::Fetch<double>(conf, myTargetInfo->Time(), MaxThetaELevel, param("CAPE-JKG"),
		                                      myTargetInfo->ForecastType());
		auto muLPLInfo = cuda::Fetch<double>(conf, myTargetInfo->Time(), MaxThetaELevel, param("LPL-M"),
		                                     myTargetInfo->ForecastType());
		auto mlCAPEInfo = cuda::Fetch<double>(conf, myTargetInfo->Time(), HalfKMLevel, param("CAPE-JKG"),
		                                      myTargetInfo->ForecastType());

		if (!muCAPEInfo || !muLPLInfo || !mlCAPEInfo)
		{
			return;
		}

		myTargetInfo->Find<param>(EBSParam);
		myTargetInfo->Find<level>(MaxWindLevel);

		const auto& muEBS = VEC(myTargetInfo);
		const auto& muLPL = VEC(muLPLInfo);
		const auto& muCAPE = VEC(muCAPEInfo);
		const auto& mlCAPE = VEC(mlCAPEInfo);

		const auto Levels = STABILITY::GetEBSLevelData(conf, myTargetInfo, h, HalfKMLevel, MaxWindLevel);

		CUDA_CHECK(cudaMalloc((void**)&d_mucape, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_mlcape, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_mlebs, memsize));

		// reusing mucape and mlcape variables
		const auto u = STABILITY::Shear(h, UParam, Levels.first, Levels.second);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_mucape, (const void*)u.data(), memsize, cudaMemcpyHostToDevice, stream));

		const auto v = STABILITY::Shear(h, VParam, Levels.first, Levels.second);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_mlcape, (const void*)v.data(), memsize, cudaMemcpyHostToDevice, stream));

		BulkShearKernel<<<gridSize, blockSize, 0, stream>>>(d_mucape, d_mlcape, d_mlebs, muEBS.size());

		CUDA_CHECK(cudaMalloc((void**)&d_mulpl, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_muebs, memsize));

		CUDA_CHECK(cudaMalloc((void**)&d_csi, memsize));

		CUDA_CHECK(
		    cudaMemcpyAsync((void*)d_mucape, (const void*)muCAPE.data(), memsize, cudaMemcpyHostToDevice, stream));

		CUDA_CHECK(
		    cudaMemcpyAsync((void*)d_mlcape, (const void*)mlCAPE.data(), memsize, cudaMemcpyHostToDevice, stream));

		CUDA_CHECK(cudaMemcpyAsync((void*)d_mulpl, (const void*)muLPL.data(), memsize, cudaMemcpyHostToDevice, stream));

		CUDA_CHECK(cudaMemcpyAsync((void*)d_muebs, (const void*)muEBS.data(), memsize, cudaMemcpyHostToDevice, stream));

		CSIKernel<<<gridSize, blockSize, 0, stream>>>(d_mucape, d_mlcape, d_mulpl, d_muebs, d_mlebs, d_csi,
		                                              myTargetInfo->SizeLocations());

		myTargetInfo->Find<param>(CSIParam);
		myTargetInfo->Find<level>(Height0Level);

		cuda::ReleaseInfo(myTargetInfo, d_csi, stream);

		CUDA_CHECK(cudaFree(d_mucape));
		CUDA_CHECK(cudaFree(d_mlcape));
		CUDA_CHECK(cudaFree(d_mulpl));
		CUDA_CHECK(cudaFree(d_muebs));
		CUDA_CHECK(cudaFree(d_mlebs));

		CUDA_CHECK(cudaFree(d_csi));
	}
	catch (const himan::HPExceptionType& e)
	{
		if (e == himan::kFileDataNotFound)
		{
			return;
		}
	}
}
namespace stabilitygpu
{
void Process(std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<info<double>> myTargetInfo)
{
	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	const size_t N = myTargetInfo->SizeLocations();

	memsize = N * sizeof(double);
	blockSize = 512;
	gridSize = N / blockSize + (N % blockSize == 0 ? 0 : 1);

	auto h = GET_PLUGIN(hitool);
	h->Configuration(conf);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());

	/* =====================================
	 * |                                   |
	 * |        LIFTED INDICES             |
	 * |                                   |
	 * =====================================
	 */

	CalculateLiftedIndices(conf, myTargetInfo, h, stream);

	/* =====================================
	 * |                                   |
	 * |            THETAE                 |
	 * |                                   |
	 * =====================================
	 */

	CalculateThetaEIndices(myTargetInfo, h, stream);

	/* =====================================
	 * |                                   |
	 * |          BULK SHEAR               |
	 * |                                   |
	 * =====================================
	 */

	CalculateBulkShear(conf, myTargetInfo, h, stream);

	/* =====================================
	 * |                                   |
	 * |            HELICITY               |
	 * |                                   |
	 * =====================================
	 */

	CalculateHelicity(conf, myTargetInfo, h, stream);

	/* =====================================
	 * |                                   |
	 * |       BULK-RICHARDSON NUMBER      |
	 * |                                   |
	 * =====================================
	 */

	CalculateBulkRichardsonNumber(conf, myTargetInfo, h, stream);

	/* =====================================
	 * |                                   |
	 * |       CONVECTIVE SEVERITY INDEX   |
	 * |                                   |
	 * =====================================
	 */

	CalculateConvectiveSeverityIndex(conf, myTargetInfo, h, stream);

	// FINISHED

	CUDA_CHECK(cudaStreamDestroy(stream));
}
}
