#include "cuda_plugin_helper.h"
#include "metutil.h"
#include "stability.cuh"
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/execution_policy.h>

#define HIMAN_AUXILIARY_INCLUDE
#include "hitool.h"
#undef HIMAN_AUXILIARY_INCLUDE

himan::level himan::plugin::stability_cuda::itsBottomLevel;

namespace STABILITY
{
extern std::vector<double> Shear(std::shared_ptr<himan::plugin::hitool>& h, const himan::param& par, double lowerHeight,
                                 double upperHeight, size_t N);
std::vector<double> Shear(std::shared_ptr<himan::plugin::hitool>& h, const himan::param& par,
                          const std::vector<double>& lowerHeight, const std::vector<double>& upperHeight);

extern himan::info_t Fetch(std::shared_ptr<const himan::plugin_configuration>& conf,
                           std::shared_ptr<himan::info>& myTargetInfo, const himan::level& lev,
                           const himan::param& par, bool returnPacked = true);
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

__global__ void StaticIndicesKernel(cdarr_t d_t850, cdarr_t d_t700, cdarr_t d_t500, cdarr_t d_td850, cdarr_t d_td700,
                                    darr_t d_ki, darr_t d_vti, darr_t d_cti, darr_t d_tti,
                                    himan::plugin::stability_cuda::options opts)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		const double T850 = d_t850[idx];
		const double T700 = d_t700[idx];
		const double T500 = d_t500[idx];
		const double TD850 = d_td850[idx];
		const double TD700 = d_td700[idx];

		d_ki[idx] = STABILITY::KI(T850, T700, T500, TD850, TD700);
		d_cti[idx] = STABILITY::CTI(T500, TD850);
		d_vti[idx] = STABILITY::VTI(T850, T500);
		d_tti[idx] = STABILITY::TTI(T850, T500, TD850);
	}
}

__global__ void ThetaEKernel(cdarr_t d_tstart, cdarr_t d_rhstart, cdarr_t d_pstart, cdarr_t d_tstop, cdarr_t d_rhstop,
                             cdarr_t d_pstop, darr_t d_thetaediff, himan::plugin::stability_cuda::options opts)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
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

__global__ void DynamicIndicesKernel(cdarr_t d_t850, cdarr_t d_t500, cdarr_t d_t500m, cdarr_t d_td850, cdarr_t d_td500m,
                                     cdarr_t d_p500m, darr_t d_si, darr_t d_li,
                                     himan::plugin::stability_cuda::options opts)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		const double t_li = himan::metutil::Lift_(d_p500m[idx], d_t500m[idx], d_td500m[idx], 50000.);
		const double t_si = himan::metutil::Lift_(85000., d_t850[idx], d_td850[idx], 50000.);

		d_si[idx] = d_t500[idx] - t_si;
		d_li[idx] = d_t500[idx] - t_li;
	}
}

__global__ void BulkShearKernel(cdarr_t d_u, cdarr_t d_v, darr_t d_bs, himan::plugin::stability_cuda::options opts)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		const double u = d_u[idx];
		const double v = d_v[idx];

		d_bs[idx] = __dsqrt_rn(u * u + v * v);
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
			u = himan::numerical_functions::interpolation::Linear(stopHeight, pz, z, pu, u);
			v = himan::numerical_functions::interpolation::Linear(stopHeight, pz, z, pv, v);

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

void CalculateBulkShear(himan::plugin::stability_cuda::options& opts, cudaStream_t& stream)
{
	using namespace himan;

	double* d_bs01 = 0;
	double* d_bs03 = 0;
	double* d_bs06 = 0;
	double* d_ebs = 0;
	double* d_u01 = 0;
	double* d_v01 = 0;
	double* d_u03 = 0;
	double* d_v03 = 0;
	double* d_u06 = 0;
	double* d_v06 = 0;
	double* d_uebs = 0;
	double* d_vebs = 0;
	double* d_el = 0;
	double* d_lpl = 0;

	try
	{
		CUDA_CHECK(cudaMalloc((void**)&d_bs01, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_bs03, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_bs06, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_ebs, memsize));

		PrepareInfo(opts.bs01);
		PrepareInfo(opts.bs03);
		PrepareInfo(opts.bs06);
		PrepareInfo(opts.ebs);

		CUDA_CHECK(cudaMalloc((void**)&d_u01, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_v01, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_u03, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_v03, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_u06, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_v06, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_uebs, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_vebs, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_el, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_lpl, memsize));

		const auto u01 = STABILITY::Shear(opts.h, UParam, 10, 1000, opts.N);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_u01, (const void*)u01.data(), memsize, cudaMemcpyHostToDevice, stream));

		const auto v01 = STABILITY::Shear(opts.h, VParam, 10, 1000, opts.N);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_v01, (const void*)v01.data(), memsize, cudaMemcpyHostToDevice, stream));

		BulkShearKernel<<<gridSize, blockSize, 0, stream>>>(d_u01, d_v01, d_bs01, opts);

		himan::ReleaseInfo(opts.bs01, d_bs01, stream);

		const auto u03 = STABILITY::Shear(opts.h, UParam, 10, 3000, opts.N);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_u03, (const void*)u03.data(), memsize, cudaMemcpyHostToDevice, stream));

		const auto v03 = STABILITY::Shear(opts.h, VParam, 10, 3000, opts.N);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_v03, (const void*)v03.data(), memsize, cudaMemcpyHostToDevice, stream));

		BulkShearKernel<<<gridSize, blockSize, 0, stream>>>(d_u03, d_v03, d_bs03, opts);

		himan::ReleaseInfo(opts.bs03, d_bs03, stream);

		const auto u06 = STABILITY::Shear(opts.h, UParam, 10, 6000, opts.N);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_u06, (const void*)u06.data(), memsize, cudaMemcpyHostToDevice, stream));

		const auto v06 = STABILITY::Shear(opts.h, VParam, 10, 6000, opts.N);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_v06, (const void*)v06.data(), memsize, cudaMemcpyHostToDevice, stream));

		BulkShearKernel<<<gridSize, blockSize, 0, stream>>>(d_u06, d_v06, d_bs06, opts);

		himan::ReleaseInfo(opts.bs06, d_bs06, stream);

		auto ELInfo = STABILITY::Fetch(opts.conf, opts.myTargetInfo, level(kMaximumThetaE, 0), param("EL-LAST-M"), false);
		auto LPLInfo = STABILITY::Fetch(opts.conf, opts.myTargetInfo, level(kMaximumThetaE, 0), param("LPL-M"), false);

		const auto& el = VEC(ELInfo);
		const auto& lpl = VEC(LPLInfo);

		std::vector<double> Midway(el.size());

		for (size_t i = 0; i < el.size(); i++)
		{
			Midway[i] = 0.5 * (el[i] - lpl[i]) + lpl[i];
		}

		const auto uebs = STABILITY::Shear(opts.h, UParam, lpl, Midway);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_uebs, (const void*)uebs.data(), memsize, cudaMemcpyHostToDevice, stream));

		const auto vebs = STABILITY::Shear(opts.h, VParam, lpl, Midway);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_vebs, (const void*)vebs.data(), memsize, cudaMemcpyHostToDevice, stream));

		BulkShearKernel<<<gridSize, blockSize, 0, stream>>>(d_uebs, d_vebs, d_ebs, opts);

		himan::ReleaseInfo(opts.ebs, d_ebs, stream);

		CUDA_CHECK(cudaStreamSynchronize(stream));

		CUDA_CHECK(cudaFree(d_bs01));
		CUDA_CHECK(cudaFree(d_bs03));
		CUDA_CHECK(cudaFree(d_bs06));
		CUDA_CHECK(cudaFree(d_ebs));
		CUDA_CHECK(cudaFree(d_u01));
		CUDA_CHECK(cudaFree(d_v01));
		CUDA_CHECK(cudaFree(d_u03));
		CUDA_CHECK(cudaFree(d_v03));
		CUDA_CHECK(cudaFree(d_u06));
		CUDA_CHECK(cudaFree(d_v06));
		CUDA_CHECK(cudaFree(d_uebs));
		CUDA_CHECK(cudaFree(d_vebs));
		CUDA_CHECK(cudaFree(d_el));
		CUDA_CHECK(cudaFree(d_lpl));
	}
	catch (const himan::HPExceptionType& e)
	{
		if (e == himan::kFileDataNotFound)
		{
			if (d_bs01)
				CUDA_CHECK(cudaFree(d_bs01));
			if (d_bs03)
				CUDA_CHECK(cudaFree(d_bs03));
			if (d_bs06)
				CUDA_CHECK(cudaFree(d_bs06));
			if (d_ebs)
				CUDA_CHECK(cudaFree(d_ebs));
			if (d_u01)
				CUDA_CHECK(cudaFree(d_u01));
			if (d_v01)
				CUDA_CHECK(cudaFree(d_v01));
			if (d_u03)
				CUDA_CHECK(cudaFree(d_u03));
			if (d_v03)
				CUDA_CHECK(cudaFree(d_v03));
			if (d_u06)
				CUDA_CHECK(cudaFree(d_u06));
			if (d_v06)
				CUDA_CHECK(cudaFree(d_v06));
			if (d_uebs)
				CUDA_CHECK(cudaFree(d_uebs));
			if (d_vebs)
				CUDA_CHECK(cudaFree(d_vebs));
			if (d_el)
				CUDA_CHECK(cudaFree(d_el));
			if (d_lpl)
				CUDA_CHECK(cudaFree(d_lpl));
		}
	}
}

void StormRelativeHelicity(himan::plugin::stability_cuda::options& opts, double* d_srh, double stopHeight,
                           cudaStream_t& stream)
{
	using namespace himan;
	using himan::plugin::stability_cuda::itsBottomLevel;

	const int blockSize = 512;
	const int gridSize = opts.N / blockSize + (opts.N % blockSize == 0 ? 0 : 1);

	InitializeArray<double>(d_srh, 0, opts.N, stream);

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
		auto Uavg = opts.h->VerticalAverage(UParam, 10, 6000);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_uavg, (const void*)Uavg.data(), memsize, cudaMemcpyHostToDevice, stream));

		auto Vavg = opts.h->VerticalAverage(VParam, 10, 6000);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_vavg, (const void*)Vavg.data(), memsize, cudaMemcpyHostToDevice, stream));

		// shear
		auto Ushear = STABILITY::Shear(opts.h, UParam, 10, 6000, Uavg.size());
		CUDA_CHECK(cudaMemcpyAsync((void*)d_ushr, (const void*)Ushear.data(), memsize, cudaMemcpyHostToDevice, stream));

		auto Vshear = STABILITY::Shear(opts.h, VParam, 10, 6000, Uavg.size());
		CUDA_CHECK(cudaMemcpyAsync((void*)d_vshr, (const void*)Vshear.data(), memsize, cudaMemcpyHostToDevice, stream));

		UVIdVectorKernel<<<gridSize, blockSize, 0, stream>>>(d_uid, d_vid, d_uavg, d_vavg, d_ushr, d_vshr, opts.N);
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
		CUDA_CHECK(cudaMalloc((void**)&d_found, opts.N * sizeof(unsigned char)));

		CUDA_CHECK(cudaStreamSynchronize(stream));

		// U&V id vector source data is not needed anymore
		CUDA_CHECK(cudaFree(d_uavg));
		CUDA_CHECK(cudaFree(d_vavg));
		CUDA_CHECK(cudaFree(d_ushr));
		CUDA_CHECK(cudaFree(d_vshr));

		InitializeArray<unsigned char>(d_found, 0, opts.N, stream);

		auto prevUInfo = STABILITY::Fetch(opts.conf, opts.myTargetInfo, itsBottomLevel, UParam);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_pu, (const void*)prevUInfo->Data().ValuesAsPOD(), memsize,
		                           cudaMemcpyHostToDevice, stream));

		auto prevVInfo = STABILITY::Fetch(opts.conf, opts.myTargetInfo, itsBottomLevel, VParam);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_pv, (const void*)prevVInfo->Data().ValuesAsPOD(), memsize,
		                           cudaMemcpyHostToDevice, stream));

		auto prevZInfo = STABILITY::Fetch(opts.conf, opts.myTargetInfo, itsBottomLevel, param("HL-M"));
		CUDA_CHECK(cudaMemcpyAsync((void*)d_pz, (const void*)prevZInfo->Data().ValuesAsPOD(), memsize,
		                           cudaMemcpyHostToDevice, stream));

		thrust::device_ptr<unsigned char> dt_found = thrust::device_pointer_cast(d_found);

		level curLevel = itsBottomLevel;

		while (curLevel.Value() > 0)
		{
			curLevel.Value(curLevel.Value() - 1);

			auto UInfo = STABILITY::Fetch(opts.conf, opts.myTargetInfo, curLevel, UParam);
			CUDA_CHECK(cudaMemcpyAsync((void*)d_u, (const void*)UInfo->Data().ValuesAsPOD(), memsize,
			                           cudaMemcpyHostToDevice, stream));

			auto VInfo = STABILITY::Fetch(opts.conf, opts.myTargetInfo, curLevel, VParam);
			CUDA_CHECK(cudaMemcpyAsync((void*)d_v, (const void*)VInfo->Data().ValuesAsPOD(), memsize,
			                           cudaMemcpyHostToDevice, stream));

			auto ZInfo = STABILITY::Fetch(opts.conf, opts.myTargetInfo, curLevel, param("HL-M"));
			CUDA_CHECK(cudaMemcpyAsync((void*)d_z, (const void*)ZInfo->Data().ValuesAsPOD(), memsize,
			                           cudaMemcpyHostToDevice, stream));

			StormRelativeHelicityKernel<<<gridSize, blockSize, 0, stream>>>(d_srh, d_u, d_v, d_pu, d_pv, d_uid, d_vid,
			                                                                d_z, d_pz, d_found, stopHeight, opts.N);

			size_t foundCount = thrust::count(thrust::cuda::par.on(stream), dt_found, dt_found + opts.N, 1);
			CUDA_CHECK(cudaStreamSynchronize(stream));

			if (foundCount == opts.N)
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
			if (d_uid)
				CUDA_CHECK(cudaFree(d_uid));
			if (d_vid)
				CUDA_CHECK(cudaFree(d_vid));
			if (d_pu)
				CUDA_CHECK(cudaFree(d_pu));
			if (d_pv)
				CUDA_CHECK(cudaFree(d_pv));
			if (d_pt)
				CUDA_CHECK(cudaFree(d_pt));
			if (d_pz)
				CUDA_CHECK(cudaFree(d_pz));
			if (d_u)
				CUDA_CHECK(cudaFree(d_u));
			if (d_v)
				CUDA_CHECK(cudaFree(d_v));
			if (d_z)
				CUDA_CHECK(cudaFree(d_z));
		}
	}

	thrust::device_ptr<double> dt_srh = thrust::device_pointer_cast(d_srh);
	thrust::replace(dt_srh, dt_srh + opts.N, 0., himan::MissingDouble());
}

void EnergyHelicityIndex(himan::plugin::stability_cuda::options& opts, double* d_srh01, cudaStream_t& stream)
{
	double* d_ehi = 0;
	double* d_cape = 0;

	try
	{
		auto CAPEInfo = STABILITY::Fetch(opts.conf, opts.myTargetInfo, himan::level(himan::kHeightLayer, 500, 0),
		                                 himan::param("CAPE-JKG"));
		auto h_cape = CAPEInfo->ToSimple();

		CUDA_CHECK(cudaMalloc((void**)&d_cape, memsize));

		himan::PrepareInfo(opts.ehi);
		himan::PrepareInfo(h_cape, d_cape, stream);

		CUDA_CHECK(cudaMalloc((void**)&d_ehi, memsize));

		EHIKernel<<<gridSize, blockSize, 0, stream>>>(d_cape, d_srh01, d_ehi, opts.N);

		himan::ReleaseInfo(opts.ehi, d_ehi, stream);
		himan::ReleaseInfo(h_cape);

		CUDA_CHECK(cudaStreamSynchronize(stream));
		CUDA_CHECK(cudaFree(d_ehi));
		CUDA_CHECK(cudaFree(d_cape));
	}
	catch (const himan::HPExceptionType& e)
	{
		if (e == himan::kFileDataNotFound)
		{
		}
	}
}

void CalculateHelicity(himan::plugin::stability_cuda::options& opts, cudaStream_t& stream)
{
	double* d_srh01 = 0;
	double* d_srh03 = 0;

	himan::PrepareInfo(opts.srh01);

	CUDA_CHECK(cudaMalloc((void**)&d_srh01, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_srh03, memsize));

	StormRelativeHelicity(opts, d_srh01, 1000, stream);

	himan::ReleaseInfo(opts.srh01, d_srh01, stream);

	himan::PrepareInfo(opts.srh03);

	StormRelativeHelicity(opts, d_srh03, 3000, stream);

	himan::ReleaseInfo(opts.srh03, d_srh03, stream);

	EnergyHelicityIndex(opts, d_srh01, stream);

	CUDA_CHECK(cudaStreamSynchronize(stream));
	CUDA_CHECK(cudaFree(d_srh01));
	CUDA_CHECK(cudaFree(d_srh03));
}

void CalculateBulkRichardsonNumber(himan::plugin::stability_cuda::options& opts, cudaStream_t& stream)
{
	double* d_brn = 0;
	double* d_cape = 0;
	double* d_u6 = 0;
	double* d_v6 = 0;
	double* d_u05 = 0;
	double* d_v05 = 0;

	try
	{
		auto CAPEInfo = STABILITY::Fetch(opts.conf, opts.myTargetInfo, himan::level(himan::kHeightLayer, 500, 0),
		                                 himan::param("CAPE-JKG"));
		auto h_cape = CAPEInfo->ToSimple();

		CUDA_CHECK(cudaMalloc((void**)&d_cape, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_u6, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_v6, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_u05, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_v05, memsize));

		himan::PrepareInfo(opts.brn);
		himan::PrepareInfo(h_cape, d_cape, stream);

		auto U6 = opts.h->VerticalAverage(UParam, 10, 6000);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_u6, (const void*)U6.data(), memsize, cudaMemcpyHostToDevice, stream));

		auto V6 = opts.h->VerticalAverage(VParam, 10, 6000);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_v6, (const void*)V6.data(), memsize, cudaMemcpyHostToDevice, stream));

		auto U05 = opts.h->VerticalAverage(UParam, 10, 500);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_u05, (const void*)U05.data(), memsize, cudaMemcpyHostToDevice, stream));

		auto V05 = opts.h->VerticalAverage(VParam, 10, 500);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_v05, (const void*)V05.data(), memsize, cudaMemcpyHostToDevice, stream));

		CUDA_CHECK(cudaMalloc((void**)&d_brn, memsize));

		BRNKernel<<<gridSize, blockSize, 0, stream>>>(d_cape, d_u6, d_v6, d_u05, d_v05, d_brn, opts.N);

		himan::ReleaseInfo(opts.brn, d_brn, stream);
		himan::ReleaseInfo(h_cape);

		CUDA_CHECK(cudaStreamSynchronize(stream));
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

void CalculateDynamicIndices(himan::plugin::stability_cuda::options& opts, cdarr_t d_t850, cdarr_t d_t700,
                             cdarr_t d_t500, cdarr_t d_td850, cdarr_t d_td500, cudaStream_t& stream)
{
	double* d_si = 0;
	double* d_li = 0;
	double* d_t500m = 0;
	double* d_td500m = 0;
	double* d_p500m = 0;

	CUDA_CHECK(cudaMalloc((void**)&d_si, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_li, memsize));

	PrepareInfo(opts.li);
	PrepareInfo(opts.si);

	CUDA_CHECK(cudaMalloc((void**)&d_t500m, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_td500m, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_p500m, memsize));

	auto T500m = opts.h->VerticalAverage(TParam, 0, 500.);
	CUDA_CHECK(cudaMemcpyAsync((void*)d_t500m, (const void*)T500m.data(), memsize, cudaMemcpyHostToDevice, stream));

	auto P500m = opts.h->VerticalAverage(PParam, 0., 500.);
	CUDA_CHECK(cudaMemcpyAsync((void*)d_p500m, (const void*)P500m.data(), memsize, cudaMemcpyHostToDevice, stream));

	std::vector<double> TD500m;

	try
	{
		TD500m = opts.h->VerticalAverage(himan::param("TD-K"), 0, 500.);
		CUDA_CHECK(
		    cudaMemcpyAsync((void*)d_td500m, (const void*)TD500m.data(), memsize, cudaMemcpyHostToDevice, stream));
	}
	catch (const himan::HPExceptionType& e)
	{
		if (e == himan::kFileDataNotFound)
		{
			try
			{
				TD500m = opts.h->VerticalAverage(RHParam, 0, 500.);
				CUDA_CHECK(cudaMemcpyAsync((void*)d_td500m, (const void*)TD500m.data(), memsize, cudaMemcpyHostToDevice,
				                           stream));

				RHToTDKernel<<<gridSize, blockSize, 0, stream>>>(d_t500m, d_td500m, TD500m.size());
			}
			catch (const himan::HPExceptionType& e)
			{
				if (e == himan::kFileDataNotFound)
				{
					CUDA_CHECK(cudaFree(d_li));
					CUDA_CHECK(cudaFree(d_si));
					CUDA_CHECK(cudaFree(d_t500m));
					CUDA_CHECK(cudaFree(d_td500m));
					CUDA_CHECK(cudaFree(d_p500m));

					return;
				}
			}
		}
	}

	if (P500m[0] < 1500)
	{
		MultiplyWith<double>(d_p500m, 100, opts.N, stream);
	}

	DynamicIndicesKernel<<<gridSize, blockSize, 0, stream>>>(d_t850, d_t500, d_t500m, d_td850, d_td500m, d_p500m, d_si,
	                                                         d_li, opts);

	himan::ReleaseInfo(opts.li, d_li, stream);
	himan::ReleaseInfo(opts.si, d_si, stream);

	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaFree(d_li));
	CUDA_CHECK(cudaFree(d_si));
	CUDA_CHECK(cudaFree(d_t500m));
	CUDA_CHECK(cudaFree(d_td500m));
	CUDA_CHECK(cudaFree(d_p500m));
}

void CalculateThetaEIndices(himan::plugin::stability_cuda::options& opts, cudaStream_t& stream)
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

		auto T3000 = opts.h->VerticalValue(TParam, 3000.);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_tstop, (const void*)T3000.data(), memsize, cudaMemcpyHostToDevice, stream));

		auto RH3000 = opts.h->VerticalValue(RHParam, 3000.);
		CUDA_CHECK(
		    cudaMemcpyAsync((void*)d_rhstop, (const void*)RH3000.data(), memsize, cudaMemcpyHostToDevice, stream));

		auto P3000 = opts.h->VerticalValue(PParam, 3000.);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_pstop, (const void*)P3000.data(), memsize, cudaMemcpyHostToDevice, stream));

		auto T2 = opts.h->VerticalValue(TParam, 2.);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_tstart, (const void*)T2.data(), memsize, cudaMemcpyHostToDevice, stream));

		auto RH2 = opts.h->VerticalValue(RHParam, 2.);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_rhstart, (const void*)RH2.data(), memsize, cudaMemcpyHostToDevice, stream));

		auto P2 = opts.h->VerticalValue(PParam, 2.);
		CUDA_CHECK(cudaMemcpyAsync((void*)d_pstart, (const void*)P2.data(), memsize, cudaMemcpyHostToDevice, stream));

		PrepareInfo(opts.thetae3);

		ThetaEKernel<<<gridSize, blockSize, 0, stream>>>(d_tstart, d_rhstart, d_pstart, d_tstop, d_rhstop, d_pstop,
		                                                 d_thetaediff, opts);

		himan::ReleaseInfo(opts.thetae3, d_thetaediff, stream);

		CUDA_CHECK(cudaStreamSynchronize(stream));

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

void CalculateBasicIndices(himan::plugin::stability_cuda::options& opts, cudaStream_t& stream)
{
	double* d_t500 = 0;
	double* d_t700 = 0;
	double* d_t850 = 0;
	double* d_td700 = 0;
	double* d_td850 = 0;
	double* d_ki = 0;
	double* d_vti = 0;
	double* d_cti = 0;
	double* d_tti = 0;

	try
	{
		auto T850Info = STABILITY::Fetch(opts.conf, opts.myTargetInfo, P850Level, TParam);
		auto T700Info = STABILITY::Fetch(opts.conf, opts.myTargetInfo, P700Level, TParam);
		auto T500Info = STABILITY::Fetch(opts.conf, opts.myTargetInfo, P500Level, TParam);
		auto TD850Info = STABILITY::Fetch(opts.conf, opts.myTargetInfo, P850Level, TDParam);
		auto TD700Info = STABILITY::Fetch(opts.conf, opts.myTargetInfo, P700Level, TDParam);

		CUDA_CHECK(cudaMalloc((void**)&d_ki, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_vti, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_cti, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_tti, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_t500, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_t700, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_t850, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_td700, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_td850, memsize));

		auto h_t850 = T850Info->ToSimple();
		auto h_t700 = T700Info->ToSimple();
		auto h_t500 = T500Info->ToSimple();
		auto h_td850 = TD850Info->ToSimple();
		auto h_td700 = TD700Info->ToSimple();

		PrepareInfo(h_t500, d_t500, stream);
		PrepareInfo(h_t700, d_t700, stream);
		PrepareInfo(h_t850, d_t850, stream);
		PrepareInfo(h_td700, d_td700, stream);
		PrepareInfo(h_td850, d_td850, stream);

		PrepareInfo(opts.ki);
		PrepareInfo(opts.tti);
		PrepareInfo(opts.cti);
		PrepareInfo(opts.vti);

		StaticIndicesKernel<<<gridSize, blockSize, 0, stream>>>(d_t850, d_t700, d_t500, d_td850, d_td700, d_ki, d_vti,
		                                                        d_cti, d_tti, opts);

		himan::ReleaseInfo(opts.ki, d_ki, stream);
		himan::ReleaseInfo(opts.cti, d_cti, stream);
		himan::ReleaseInfo(opts.tti, d_tti, stream);
		himan::ReleaseInfo(opts.vti, d_vti, stream);
		CUDA_CHECK(cudaStreamSynchronize(stream));

		CUDA_CHECK(cudaFree(d_ki));
		CUDA_CHECK(cudaFree(d_vti));
		CUDA_CHECK(cudaFree(d_cti));
		CUDA_CHECK(cudaFree(d_tti));

		/* =====================================
		 * |                                   |
		 * |       DYNAMIC INDICES             |
		 * |                                   |
		 * =====================================
		*/

		CalculateDynamicIndices(opts, d_t850, d_t700, d_t500, d_td850, d_td700, stream);

		himan::ReleaseInfo(h_t500);
		himan::ReleaseInfo(h_t700);
		himan::ReleaseInfo(h_t850);
		himan::ReleaseInfo(h_td700);
		himan::ReleaseInfo(h_td850);
		CUDA_CHECK(cudaStreamSynchronize(stream));

		CUDA_CHECK(cudaFree(d_t850));
		CUDA_CHECK(cudaFree(d_t700));
		CUDA_CHECK(cudaFree(d_t500));
		CUDA_CHECK(cudaFree(d_td850));
		CUDA_CHECK(cudaFree(d_td700));
	}
	catch (const himan::HPExceptionType& e)
	{
		if (e == himan::kFileDataNotFound)
		{
			return;
		}
	}
}

void himan::plugin::stability_cuda::Process(options& opts)
{
	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	memsize = opts.N * sizeof(double);
	blockSize = 512;
	gridSize = opts.N / blockSize + (opts.N % blockSize == 0 ? 0 : 1);

	/* =====================================
	 * |                                   |
	 * |         BASIC INDICES             |
	 * |                                   |
	 * =====================================
	*/

	CalculateBasicIndices(opts, stream);

	/* =====================================
	 * |                                   |
	 * |            THETAE                 |
	 * |                                   |
	 * =====================================
	*/

	CalculateThetaEIndices(opts, stream);

	/* =====================================
	 * |                                   |
	 * |          BULK SHEAR               |
	 * |                                   |
	 * =====================================
	*/

	CalculateBulkShear(opts, stream);

	/* =====================================
	 * |                                   |
	 * |            HELICITY               |
	 * |                                   |
	 * =====================================
	*/

	CalculateHelicity(opts, stream);

	/* =====================================
	 * |                                   |
	 * |       BULK-RICHARDSON NUMBER      |
	 * |                                   |
	 * =====================================
	*/

	CalculateBulkRichardsonNumber(opts, stream);

	// FINISHED

	CUDA_CHECK(cudaStreamSynchronize(stream));

	// Free device memory

	CUDA_CHECK(cudaStreamDestroy(stream));
}
