// System includes
#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>

#include "plugin_factory.h"

#include "si_cuda.h"
#include "cuda_helper.h"
#include "metutil.h"
#include "util.h"

#include <NFmiGribPacking.h>

#include "regular_grid.h"
#include "forecast_time.h"
#include "level.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"
#include "cache.h"
#include "hitool.h"


using namespace himan;
using namespace himan::plugin;

level si_cuda::itsBottomLevel;

const unsigned char FCAPE		= (1 << 2);
const unsigned char FCAPE3km	= (1 << 0);

extern double Max(const std::vector<double>& vec);

template <typename T>
__global__ void InitializeArrayKernel(T* d_arr, T val, size_t N)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for(; idx < N; idx += stride)
	{
		d_arr[idx] = val;
	}
}

template <typename T>
void InitializeArray(T* d_arr, T val, size_t N, cudaStream_t& stream)
{
	const int blockSize = 128;
	const int gridSize = N/blockSize + (N%blockSize == 0?0:1);

	InitializeArrayKernel<T> <<< gridSize, blockSize, 0, stream >>> (d_arr, val, N);

}

template <typename T>
__global__ void MultiplyWith(T* d_arr, T val, size_t N)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for(; idx < N; idx += stride)
	{
		d_arr[idx] = d_arr[idx] * val;
	}
}

template <typename T>
void MultiplyWith(T* d_arr, T val, size_t N, cudaStream_t& stream)
{
	const int blockSize = 128;
	const int gridSize = N/blockSize + (N%blockSize == 0?0:1);

	MultiplyWith<T> <<< gridSize, blockSize, 0, stream >>> (d_arr, val, N);

}

info_simple* PrepareInfo(std::shared_ptr<himan::info> fullInfo, cudaStream_t& stream)
{
	auto h_info = fullInfo->ToSimple();
	size_t N = h_info->size_x * h_info->size_y;
	
	assert(N > 0);

	// 1. Reserve memory at device for unpacked data
	double* d_arr = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<double**> (&d_arr), N * sizeof(double)));

	// 2. Unpack if needed, leave data to device and simultaneously copy it back to cpu (himan cache)
	auto tempGrid = dynamic_cast<himan::regular_grid*> (fullInfo->Grid());

	if (tempGrid->IsPackedData())
	{
		assert(tempGrid->PackedData().ClassName() == "simple_packed" || tempGrid->PackedData().ClassName() == "jpeg_packed");
		assert(N > 0);
		assert(tempGrid->Data().Size() == N);

		double* arr = const_cast<double*> (tempGrid->Data().ValuesAsPOD());
		CUDA_CHECK(cudaHostRegister(reinterpret_cast<void*> (arr), sizeof(double) * N, 0));

		assert(arr);

		tempGrid->PackedData().Unpack(d_arr, N, &stream);

		CUDA_CHECK(cudaMemcpyAsync(arr, d_arr, sizeof(double) * N, cudaMemcpyDeviceToHost, stream));

		tempGrid->PackedData().Clear();

		auto c = GET_PLUGIN(cache);

		CUDA_CHECK(cudaStreamSynchronize(stream));

		c->Insert(*fullInfo);	

		CUDA_CHECK(cudaHostUnregister(arr));

		h_info->packed_values = 0;
	}
	else
	{
		CUDA_CHECK(cudaMemcpyAsync(d_arr, fullInfo->Data().ValuesAsPOD(), sizeof(double) * N, cudaMemcpyHostToDevice, stream));
	}

	h_info->values = d_arr;
	
	return h_info;
}

std::shared_ptr<himan::info> Fetch(const std::shared_ptr<const plugin_configuration> conf, const himan::forecast_time& theTime, const himan::level& theLevel, const himan::param& theParam, const himan::forecast_type& theType)
{
	try
	{
		auto f = GET_PLUGIN(fetcher);
		return f->Fetch(conf, theTime, theLevel, theParam, theType, true);
	}
	catch (HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw std::runtime_error("si_cuda::Fetch(): Unable to proceed");
		}
		
		return std::shared_ptr<info> ();
	}
}

__global__
void CopyLFCIteratorValuesKernel(double* __restrict__ d_Titer, const double* __restrict__ d_Tparcel, double* __restrict__ d_Piter, info_simple d_Penv)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < d_Penv.size_x * d_Penv.size_y)
	{
		if (d_Tparcel[idx] != kFloatMissing && d_Penv.values[idx] != kFloatMissing)
		{
			d_Titer[idx] = d_Tparcel[idx];
			d_Piter[idx] = d_Penv.values[idx];
		}
	}
}

__global__
void LiftLCLKernel(const double* __restrict__ d_P, const double* __restrict__ d_T, const double* __restrict__ d_PLCL, info_simple d_Ptarget, double* __restrict__ d_Tparcel)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < d_Ptarget.size_x * d_Ptarget.size_y)
	{
		assert(d_P[idx] > 10);
		assert(d_P[idx] < 1500 || d_P[idx] == kFloatMissing);

		assert(d_Ptarget.values[idx] > 10);
		assert(d_Ptarget.values[idx] < 1500 || d_Ptarget.values[idx] == kFloatMissing);

		assert(d_T[idx] > 100);
		assert(d_T[idx] < 350 || d_T[idx] == kFloatMissing);

		double T = metutil::LiftLCL_(d_P[idx]*100, d_T[idx], d_PLCL[idx]*100, d_Ptarget.values[idx]*100);

		assert(T > 100);
		assert(T < 350 || T == kFloatMissing);

		d_Tparcel[idx] = T;
	}
}

__global__
void MoistLiftKernel(const double* __restrict__ d_T, const double* __restrict__  d_P, info_simple d_Ptarget, double* __restrict__ d_Tparcel)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	assert(d_T);
	assert(d_P);

	if (idx < d_Ptarget.size_x * d_Ptarget.size_y)
	{
		assert(d_P[idx] > 10);
		assert(d_P[idx] < 1500 || d_P[idx] == kFloatMissing);

		assert(d_Ptarget.values[idx] > 10);
		assert(d_Ptarget.values[idx] < 1500 || d_Ptarget.values[idx] == kFloatMissing);

		assert(d_T[idx] > 100);
		assert(d_T[idx] < 350 || d_T[idx] == kFloatMissing);

		double T = metutil::MoistLift_(d_P[idx]*100, d_T[idx], d_Ptarget.values[idx]*100);

		assert(T > 100);
		assert(T < 350 || T == kFloatMissing);

		d_Tparcel[idx] = T;
	}
}

__global__
void CAPEKernel(info_simple d_Tenv, info_simple d_Penv, info_simple d_Zenv, info_simple d_prevTenv, info_simple d_prevPenv, info_simple d_prevZenv, 
		const double* __restrict d_Tparcel, const double* __restrict d_prevTparcel, const double* __restrict__ d_LFCP, double* __restrict__ d_CAPE, 
		double* __restrict__ d_CAPE1040, double* __restrict__ d_CAPE3km, double* __restrict__ d_ELT, double* __restrict__ d_ELP, unsigned char* __restrict__ d_found, int d_curLevel, int d_breakLevel)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < d_Tenv.size_x * d_Tenv.size_y && d_found[idx] != 4)
	{
		double Tenv = d_Tenv.values[idx];
		assert(Tenv > 100.);

		double Penv = d_Penv.values[idx]; // hPa
		assert(Penv < 1200.);
		
		double Zenv = d_Zenv.values[idx]; // m
		
		double prevTenv = d_prevTenv.values[idx]; // K
		assert(prevTenv > 100.);

		double prevPenv = d_prevPenv.values[idx]; // hPa
		assert(prevPenv < 1200.);
		
		double prevZenv = d_prevZenv.values[idx]; // m

		double Tparcel = d_Tparcel[idx]; // K
		assert(Tparcel > 100. || Tparcel == kFloatMissing);

		double prevTparcel = d_prevTparcel[idx]; // K
		assert(prevTparcel > 100. || Tparcel == kFloatMissing);

		double LFCP = d_LFCP[idx]; // hPa
		assert(LFCP < 1200.);
				
		if (Penv == kFloatMissing || Tenv == kFloatMissing || Zenv == kFloatMissing || prevZenv == kFloatMissing || Tparcel == kFloatMissing || prevTparcel == kFloatMissing || Penv > LFCP)
		{
			// Missing data or current grid point is below LFC
			;
		}
		else if (d_curLevel < d_breakLevel && (Tenv - Tparcel) > 25.)
		{
			// Temperature gap between environment and parcel too large --> abort search.
			// Only for values higher in the atmosphere, to avoid the effects of inversion

			d_found[idx] |= FCAPE;
		}
		else
		{
			if (prevZenv >= 3000. && Zenv >= 3000.)
			{
				d_found[idx] |= FCAPE3km;
			}

			if ((d_found[idx] & FCAPE3km) == 0)
			{
				double C = CAPE::CalcCAPE3km(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv);

				d_CAPE3km[idx] += C;

				assert(d_CAPE3km[idx] < 3000.); // 3000J/kg, not 3000m
				assert(d_CAPE3km[idx] >= 0);
			}

			double C = CAPE::CalcCAPE1040(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv);

			d_CAPE1040[idx] += C;

			assert(d_CAPE1040[idx] < 5000.);
			assert(d_CAPE1040[idx] >= 0);

			double CAPE, ELT, ELP;
			CAPE::CalcCAPE(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv, CAPE, ELT, ELP);

			d_CAPE[idx] += CAPE;

			assert(CAPE >= 0.);				
			assert(d_CAPE[idx] < 8000);

			if (ELT != kFloatMissing)
			{
				d_ELT[idx] = ELT;
				d_ELP[idx] = ELP;
			}
		}
	}
}

__global__
void CINKernel(info_simple d_Tenv, info_simple d_Penv, const double* __restrict__ d_Titer, const double* __restrict__ d_Piter, info_simple d_Zenv, info_simple d_prevZenv, const double* __restrict__ d_Tparcel, const double* __restrict__ d_PLCL, const double* __restrict__ d_PLFC, double* __restrict__ d_cinh, unsigned char* __restrict__ d_found)
{
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < d_Tenv.size_x * d_Tenv.size_y && d_found[idx] == 0)
	{
		
		double Tenv = d_Tenv.values[idx]; // K
		assert(Tenv >= 150.);

		double Penv = d_Penv.values[idx]; // hPa
		assert(Penv < 1200. || Penv == kFloatMissing);

		double Pparcel = d_Piter[idx]; // hPa
		assert(Pparcel < 1200. || Pparcel == kFloatMissing);

		double Tparcel = d_Titer[idx]; // K
		assert(Tparcel >= 150. || Tparcel == kFloatMissing);

		double PLFC = d_PLFC[idx]; // hPa
		assert(PLFC < 1200. || PLFC == kFloatMissing);
		
		double PLCL = d_PLCL[idx]; // hPa
		assert(PLCL < 1200. || PLCL == kFloatMissing);
		
		double Zenv = d_Zenv.values[idx]; // m
		double prevZenv = d_prevZenv.values[idx]; // m

		if (
				PLFC == kFloatMissing || // No LFC
				Penv <= PLFC // reached max height; TODO: final piece integration
				)
		{
			d_found[idx] = 1;
		}
		else
		{
			if (Penv < PLCL)
			{
				// Above LCL, switch to virtual temperature
				
				Tparcel = metutil::VirtualTemperature_(Tparcel, Penv * 100);
				Tenv = metutil::VirtualTemperature_(Tenv, Penv * 100);
			}

			if (Tparcel != kFloatMissing && Tparcel <= Tenv)
			{
				d_cinh[idx] += constants::kG * (Zenv - prevZenv) * ((Tparcel - Tenv) / Tenv);
				assert(d_cinh[idx] <= 0);
			}
			else if (d_cinh[idx] != 0)
			{
				// Parcel buoyant --> cape layer, no more CIN. We stop integration here.
				// TODO: final piece integration
				d_found[idx] = 1;
			}
		}	
	}
}

__global__
void LFCKernel(info_simple d_T, info_simple d_P, info_simple d_prevT, info_simple d_prevP, double* __restrict__ d_Tparcel, double* __restrict__ d_LCLP, double* __restrict__ d_LFCT, double* __restrict__ d_LFCP, unsigned char* __restrict__ d_found, int d_curLevel, int d_breakLevel)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	assert(d_T.values);
	assert(d_P.values);

	if (idx < d_T.size_x * d_T.size_y && d_found[idx] == 0)
	{
		double Tparcel = d_Tparcel[idx];
		double Tenv = d_T.values[idx];
		
		assert(Tenv < 350.);
		assert(Tenv > 100.);
		
		double prevTenv = d_prevT.values[idx];
		assert(prevTenv < 350.);
		assert(prevTenv > 100.);

		double Penv = d_P.values[idx];
		double LCLP = d_LCLP[idx];
		
		if (Tparcel != kFloatMissing && d_curLevel < d_breakLevel && (Tenv - Tparcel) > 30.)
		{
			// Temperature gap between environment and parcel too large --> abort search.
			// Only for values higher in the atmosphere, to avoid the effects of inversion

			d_found[idx] = 1;
		}
		
		if (Tparcel != kFloatMissing && Penv <= LCLP)
		{
			if (Tparcel >= Tenv && d_found[idx] == 0)
			{
				d_found[idx] = 1;

				// We have no specific information on the precise height where the temperature has crossed
				// Or we could if we'd integrate it but it makes the calculation more complex. So maybe in the
				// future. For now just take an average of upper and lower level values.
				
				if (prevTenv == kFloatMissing) prevTenv = Tenv;

				d_LFCT[idx] = (Tenv + prevTenv) * 0.5; // K

				assert(d_LFCT[idx] > 100);
				assert(d_LFCT[idx] < 350);

				// Never allow LFC pressure to be bigger than LCL pressure; bound lower level (with larger pressure value)
				// to LCL level if it below LCL

				double prevPenv = d_prevP.values[idx];
				prevPenv = min(prevPenv, LCLP);
				assert(prevPenv > 10);
				assert(prevPenv < 1500);

				d_LFCP[idx] = (Penv + prevPenv) * 0.5; // hPa
			}
		}
	}
}


__global__
void ThetaEKernel(info_simple d_T, info_simple d_RH, info_simple d_P, info_simple d_prevT, info_simple d_prevRH, info_simple d_prevP, double* __restrict__ d_maxThetaE, double* __restrict__ d_Tresult, double* __restrict__ d_TDresult, unsigned char* __restrict__ d_found)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	assert(d_T.values);
	assert(d_RH.values);
	assert(d_P.values);
	
	if (idx < d_T.size_x * d_T.size_y && d_found[idx] == 0)
	{
		double T = d_T.values[idx];
		double P = d_P.values[idx];
		double RH = d_RH.values[idx];
		
		if (P == kFloatMissing || T == kFloatMissing || RH == kFloatMissing)
		{
			d_found[idx] = 1;
		}
		else
		{
			if (P < 600.)
			{
				// Cut search if reach level 600hPa

				// Linearly interpolate temperature and humidity values to 600hPa, to check
				// if highest theta e is found there

				T = numerical_functions::interpolation::Linear(600., P, d_prevP.values[idx], T, d_prevT.values[idx]);
				RH = numerical_functions::interpolation::Linear(600., P, d_prevP.values[idx], RH, d_prevRH.values[idx]);

				d_found[idx] = 1; // Make sure this is the last time we access this grid point
				P = 600.;
			}

			double TD = metutil::DewPointFromRH_(T, RH);

			double& refThetaE = d_maxThetaE[idx];
			double ThetaE = metutil::ThetaE_(T, TD, P*100);

			if (ThetaE >= refThetaE)
			{
				refThetaE = ThetaE;
				d_Tresult[idx] = T;
				d_TDresult[idx] = TD;
			}
		}
	}
}

std::pair<std::vector<double>,std::vector<double>> si_cuda::GetHighestThetaETAndTDGPU(const std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<info> myTargetInfo)
{
	himan::level curLevel = itsBottomLevel;
	
	const size_t N = myTargetInfo->Data().Size();
	const int blockSize = 256;
	const int gridSize = N/blockSize + (N%blockSize == 0?0:1);

	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));
	
	double* d_maxThetaE = 0;
	double* d_Tresult = 0;
	double* d_TDresult = 0;
	unsigned char* d_found = 0;
	
	CUDA_CHECK(cudaMalloc((double**) &d_maxThetaE, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((double**) &d_Tresult, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((double**) &d_TDresult, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((double**) &d_found, sizeof(unsigned char) * N));

	InitializeArray<double> (d_maxThetaE, -1, N, stream);
	InitializeArray<double> (d_Tresult, kFloatMissing, N, stream);
	InitializeArray<double> (d_TDresult, kFloatMissing, N, stream);
	InitializeArray<unsigned char> (d_found, 0, N, stream);
	
	info_simple* h_prevT = 0;
	info_simple* h_prevP = 0;
	info_simple* h_prevRH = 0;
	
	while (true)
	{

		auto TInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType());
		auto RHInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("RH-PRCNT"), myTargetInfo->ForecastType());
		auto PInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType());

		if (!TInfo || !RHInfo || !PInfo)
		{
			return std::make_pair(std::vector<double>(),std::vector<double>());
		}

		auto h_T = PrepareInfo(TInfo, stream);
		auto h_P = PrepareInfo(PInfo, stream);
		auto h_RH = PrepareInfo(RHInfo, stream);

		assert(h_T->values);
		assert(h_RH->values);
		assert(h_P->values);

		bool release = true;

		if (!h_prevT)
		{
			// first time
			h_prevT = new info_simple(*h_T);
			h_prevP = new info_simple(*h_P);
			h_prevRH = new info_simple(*h_RH);

			release = false;
		}

		ThetaEKernel <<< gridSize, blockSize, 0, stream >>> (*h_T, *h_RH, *h_P, *h_prevT, *h_prevRH, *h_prevP, d_maxThetaE, d_Tresult, d_TDresult, d_found);

		std::vector<unsigned char> found(N, 0);
		CUDA_CHECK(cudaMemcpyAsync(&found[0], d_found, sizeof(unsigned char) * N, cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaStreamSynchronize(stream));

		if (release)
		{
			CUDA_CHECK(cudaFree(h_prevP->values));
			CUDA_CHECK(cudaFree(h_prevRH->values));
			CUDA_CHECK(cudaFree(h_prevT->values));
		}
				
		delete h_prevP;
		delete h_prevT;
		delete h_prevRH;

		h_prevP = h_P;
		h_prevRH = h_RH;
		h_prevT = h_T;
		
		curLevel.Value(curLevel.Value()-1);

		size_t foundCount = std::count(found.begin(), found.end(), 1);

		if (foundCount == found.size()) break;
	}
	
	CUDA_CHECK(cudaFree(h_prevP->values));
	CUDA_CHECK(cudaFree(h_prevRH->values));
	CUDA_CHECK(cudaFree(h_prevT->values));
	
	delete h_prevP;
	delete h_prevT;
	delete h_prevRH;

	std::vector<double> Tsurf(myTargetInfo->Data().Size());
	std::vector<double> TDsurf(myTargetInfo->Data().Size());

	CUDA_CHECK(cudaMemcpyAsync(&Tsurf[0], d_Tresult, sizeof(double) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(&TDsurf[0], d_TDresult, sizeof(double) * N, cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaStreamSynchronize(stream));
	
	CUDA_CHECK(cudaFree(d_maxThetaE));
	CUDA_CHECK(cudaFree(d_Tresult));
	CUDA_CHECK(cudaFree(d_TDresult));
	CUDA_CHECK(cudaFree(d_found));
	
	CUDA_CHECK(cudaStreamDestroy(stream));

	return std::make_pair(Tsurf, TDsurf);
	
}

std::pair<std::vector<double>,std::vector<double>> si_cuda::GetLFCGPU(const std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<info> myTargetInfo, std::vector<double>& T, std::vector<double>& P, std::vector<double>& TenvLCL)
{
	auto h = GET_PLUGIN(hitool);
	h->Configuration(conf);
	h->Time(myTargetInfo->Time());
	h->HeightUnit(kHPa);

	const size_t N = myTargetInfo->Data().Size();
	const int blockSize = 256;
	const int gridSize = N/blockSize + (N%blockSize == 0?0:1);

	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	double* d_TenvLCL = 0;
	double* d_Titer = 0;
	double* d_Piter = 0;
	double* d_LCLP = 0;
	double* d_LFCT = 0;
	double* d_LFCP = 0;
	double* d_Tparcel = 0;

	unsigned char* d_found = 0;
	
	CUDA_CHECK(cudaMalloc((double**) &d_TenvLCL, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((double**) &d_Piter, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((double**) &d_Titer, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((double**) &d_LCLP, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((double**) &d_LFCT, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((double**) &d_LFCP, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((double**) &d_found, sizeof(unsigned char) * N));
	CUDA_CHECK(cudaMalloc((double**) &d_Tparcel, sizeof(double) * N));

	CUDA_CHECK(cudaMemcpyAsync(d_TenvLCL, &TenvLCL[0], sizeof(double) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_Titer, &T[0], sizeof(double) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_Piter, &P[0], sizeof(double) * N, cudaMemcpyHostToDevice, stream));
	
	CUDA_CHECK(cudaMemcpyAsync(d_LCLP, d_Piter, sizeof(double) * N, cudaMemcpyDeviceToDevice, stream));
	
	InitializeArray<double> (d_LFCT, kFloatMissing, N, stream);
	InitializeArray<double> (d_LFCP, kFloatMissing, N, stream);
	InitializeArray<unsigned char> (d_found, 0, N, stream);

	// For each grid point find the hybrid level that's below LCL and then pick the lowest level
	// among all grid points; most commonly it's the lowest hybrid level

	auto levels = h->LevelForHeight(myTargetInfo->Producer(), ::Max(P));

	level curLevel = levels.first;
	
	auto prevPenvInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType());
	auto prevTenvInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType());

	auto h_prevTenv = PrepareInfo(prevTenvInfo, stream);
	auto h_prevPenv = PrepareInfo(prevPenvInfo, stream);

	assert(h_prevTenv->values);
	assert(h_prevPenv->values);

	curLevel.Value(curLevel.Value()-1);

	std::vector<unsigned char> found(N, 0);
	std::vector<double> LFCT(N, kFloatMissing);
	std::vector<double> LFCP(N, kFloatMissing);

	for (size_t i = 0; i < N; i++)
	{
		if (T[i] >= TenvLCL[i])
		{
			found[i] = 1;
			LFCT[i] = T[i];
			LFCP[i] = P[i];
		}
	}

	CUDA_CHECK(cudaMemcpyAsync(d_found, &found[0], sizeof(unsigned char) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_LFCT, &LFCT[0], sizeof(double) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_LFCP, &LFCP[0], sizeof(double) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));

	auto hPa450 = h->LevelForHeight(myTargetInfo->Producer(), 450.);
	auto hPa150 = h->LevelForHeight(myTargetInfo->Producer(), 150.);

	while (curLevel.Value() > hPa150.first.Value())
	{	
		// Get environment temperature and pressure values for this level
		auto TenvInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType());
		auto PenvInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType());

		auto h_Penv = PrepareInfo(PenvInfo, stream);
		auto h_Tenv = PrepareInfo(TenvInfo, stream);

		// Lift the particle from previous level to this level. In the first revolution
		// of this loop the starting level is LCL. If target level level is below current level
		// (ie. we would be lowering the particle) missing value is returned.

		MoistLiftKernel <<< gridSize, blockSize, 0, stream >>> (d_Titer, d_Piter, *h_Penv, d_Tparcel);

		LFCKernel <<< gridSize, blockSize, 0, stream >>> (*h_Tenv, *h_Penv, *h_prevTenv, *h_prevPenv, d_Tparcel, d_LCLP, d_LFCT, d_LFCP, d_found, curLevel.Value(), hPa450.first.Value());

		CUDA_CHECK(cudaMemcpyAsync(&found[0], d_found, sizeof(unsigned char) * N, cudaMemcpyDeviceToHost, stream));

		CUDA_CHECK(cudaFree(h_prevPenv->values));
		CUDA_CHECK(cudaFree(h_prevTenv->values));

		delete h_prevPenv;
		delete h_prevTenv;

		h_prevPenv = h_Penv;
		h_prevTenv = h_Tenv;

		CUDA_CHECK(cudaStreamSynchronize(stream));

		if (static_cast<size_t> (std::count(found.begin(), found.end(), 1)) == found.size()) break;

		// preserve starting position for those grid points that have value

		CopyLFCIteratorValuesKernel <<< gridSize, blockSize, 0, stream >>> (d_Titer, d_Tparcel, d_Piter, *h_Penv);
		
		curLevel.Value(curLevel.Value() - 1);	

	}

	CUDA_CHECK(cudaFree(h_prevPenv->values));
	CUDA_CHECK(cudaFree(h_prevTenv->values));

	delete h_prevPenv;
	delete h_prevTenv;

	CUDA_CHECK(cudaMemcpyAsync(&LFCT[0], d_LFCT, sizeof(double) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(&LFCP[0], d_LFCP, sizeof(double) * N, cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaFree(d_LFCT));
	CUDA_CHECK(cudaFree(d_LFCP));
	CUDA_CHECK(cudaFree(d_LCLP));
	CUDA_CHECK(cudaFree(d_Tparcel));
	CUDA_CHECK(cudaFree(d_found));
	CUDA_CHECK(cudaFree(d_Titer));
	CUDA_CHECK(cudaFree(d_Piter));
	CUDA_CHECK(cudaFree(d_TenvLCL));

	CUDA_CHECK(cudaStreamDestroy(stream));

	return std::make_pair(LFCT, LFCP);
}


void si_cuda::GetCINGPU(const std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<info> myTargetInfo, const std::vector<double>& Tsurf, const std::vector<double>& TLCL, const std::vector<double>& PLCL, const std::vector<double>& PLFC, param CINParam)
{
	const params PParams({param("PGR-PA"), param("P-PA")});

	auto h = GET_PLUGIN(hitool);
	h->Configuration(conf);
	h->Time(myTargetInfo->Time());
	h->HeightUnit(kHPa);

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
	
	// Get LCL and LFC heights in meters

	auto ZLCL = h->VerticalValue(param("HL-M"), PLCL);
	auto ZLFC = h->VerticalValue(param("HL-M"), PLFC);

	level curLevel = itsBottomLevel;
	
	auto basePenvInfo = Fetch(conf, ftime, curLevel, param("P-HPA"), ftype);
	auto prevZenvInfo = Fetch(conf, ftime, curLevel, param("HL-M"), ftype);
	auto prevTenvInfo = Fetch(conf, ftime, curLevel, param("T-K"), ftype);
	auto prevPenvInfo = Fetch(conf, ftime, curLevel, param("P-HPA"), ftype);

	const size_t N = myTargetInfo->Data().Size();
	const int blockSize = 256;
	const int gridSize = N/blockSize + (N%blockSize == 0?0:1);

	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	auto h_basePenv = PrepareInfo(basePenvInfo, stream);
	auto h_prevZenv = PrepareInfo(prevZenvInfo, stream);
	auto h_prevTenv = PrepareInfo(prevTenvInfo, stream);
	auto h_prevPenv = PrepareInfo(prevPenvInfo, stream);
	
	double* d_Tparcel = 0;
	double* d_Piter = 0;
	double* d_Titer = 0;
	double* d_PLCL = 0;
	double* d_PLFC = 0;
	double* d_cinh = 0;
	unsigned char* d_found = 0;
	
	CUDA_CHECK(cudaMalloc((double**) &d_Tparcel, N * sizeof(double)));
	CUDA_CHECK(cudaMalloc((double**) &d_Piter, N * sizeof(double)));
	CUDA_CHECK(cudaMalloc((double**) &d_Titer, N * sizeof(double)));
	CUDA_CHECK(cudaMalloc((double**) &d_PLCL, N * sizeof(double)));
	CUDA_CHECK(cudaMalloc((double**) &d_PLFC, N * sizeof(double)));
	CUDA_CHECK(cudaMalloc((double**) &d_cinh, N * sizeof(double)));
	CUDA_CHECK(cudaMalloc((unsigned char**) &d_found, N * sizeof(unsigned char)));
	
	InitializeArray<double>(d_cinh, 0., N, stream);
	InitializeArray<double>(d_Tparcel, kFloatMissing, N, stream);
	InitializeArray<unsigned char>(d_found, 0, N, stream);

	CUDA_CHECK(cudaMemcpyAsync(d_Titer, &Tsurf[0], sizeof(double) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_Piter, h_basePenv->values, sizeof(double) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_PLCL, &PLCL[0], sizeof(double) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_PLFC, &PLFC[0], sizeof(double) * N, cudaMemcpyHostToDevice, stream));
	
	std::vector<unsigned char> found(N, 0);
	
	curLevel.Value(curLevel.Value()-1);

	auto hPa150 = h->LevelForHeight(myTargetInfo->Producer(), 150.);

	while (curLevel.Value() > hPa150.first.Value())
	{

		auto ZenvInfo = Fetch(conf, ftime, curLevel, param("HL-M"), ftype);
		auto TenvInfo = Fetch(conf, ftime, curLevel, param("T-K"), ftype);
		auto PenvInfo = Fetch(conf, ftime, curLevel, param("P-HPA"), ftype);
	
		auto h_Zenv = PrepareInfo(ZenvInfo, stream);
		auto h_Penv = PrepareInfo(PenvInfo, stream);
		auto h_Tenv = PrepareInfo(TenvInfo, stream);
		
		LiftLCLKernel <<< gridSize, blockSize, 0, stream >>> (d_Piter, d_Titer, d_PLCL, *h_Penv, d_Tparcel);
		
		CINKernel <<< gridSize, blockSize, 0, stream >>> (*h_Tenv, *h_Penv, d_Titer, d_Piter, *h_Zenv, *h_prevZenv, d_Tparcel, d_PLCL, d_PLFC, d_cinh, d_found);
		
		CUDA_CHECK(cudaMemcpyAsync(&found[0], d_found, sizeof(unsigned char) * N, cudaMemcpyDeviceToHost, stream));

		CUDA_CHECK(cudaFree(h_prevPenv->values));
		CUDA_CHECK(cudaFree(h_prevTenv->values));
		CUDA_CHECK(cudaFree(h_prevZenv->values));

		delete h_prevPenv;
		delete h_prevTenv;
		delete h_prevZenv;

		h_prevPenv = h_Penv;
		h_prevTenv = h_Tenv;
		h_prevZenv = h_Zenv;

		CUDA_CHECK(cudaStreamSynchronize(stream));

		if (static_cast<size_t> (std::count(found.begin(), found.end(), 1)) == found.size()) break;

		// preserve starting position for those grid points that have value

		CopyLFCIteratorValuesKernel <<< gridSize, blockSize, 0, stream >>> (d_Titer, d_Tparcel, d_Piter, *h_Penv);
		
		curLevel.Value(curLevel.Value() - 1);

		
	}
	
	CUDA_CHECK(cudaFree(h_prevPenv->values));
	CUDA_CHECK(cudaFree(h_prevTenv->values));
	CUDA_CHECK(cudaFree(h_prevZenv->values));

	delete h_prevPenv;
	delete h_prevTenv;
	delete h_prevZenv;

	std::vector<double> cinh(N, 0);
	
	CUDA_CHECK(cudaMemcpyAsync(&cinh[0], d_cinh, sizeof(double) * N, cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaFree(d_cinh));
	CUDA_CHECK(cudaFree(d_Tparcel));
	CUDA_CHECK(cudaFree(d_Piter));
	CUDA_CHECK(cudaFree(d_Titer));
	CUDA_CHECK(cudaFree(d_PLCL));
	CUDA_CHECK(cudaFree(d_PLFC));
	CUDA_CHECK(cudaFree(d_found));

	CUDA_CHECK(cudaStreamDestroy(stream));

	myTargetInfo->Param(CINParam);
	myTargetInfo->Data().Set(cinh);

}

void si_cuda::GetCAPEGPU(const std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<info> myTargetInfo, const std::vector<double>& T, const std::vector<double>& P, param ELTParam, param ELPParam, param CAPEParam, param CAPE1040Param, param CAPE3kmParam)
{
	assert(T.size() == P.size());

	auto h = GET_PLUGIN(hitool);
	
	h->Configuration(conf);
	h->Time(myTargetInfo->Time());
	h->HeightUnit(kHPa);

	// Found count determines if we have calculated all three CAPE variation for a single grid point
	std::vector<unsigned char> found(T.size(), 0);

	// No LFC --> No CAPE
	
	for (size_t i = 0; i < P.size(); i++)
	{
		if (P[i] == kFloatMissing)
		{
			found[i] |= FCAPE;
		}
	}
	
	const size_t N = myTargetInfo->Data().Size();
	const int blockSize = 256;
	const int gridSize = N/blockSize + (N%blockSize == 0?0:1);

	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	double* d_CAPE = 0;
	double* d_CAPE1040 = 0;
	double* d_CAPE3km = 0;
	double* d_ELT = 0;
	double* d_ELP = 0;
	double* d_Titer = 0;
	double* d_Piter = 0;
	double* d_prevTparcel = 0;
	double* d_Tparcel = 0;
	double* d_LFC = 0;
	
	unsigned char* d_found = 0;
	
	CUDA_CHECK(cudaMalloc((double**) &d_CAPE, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((double**) &d_CAPE1040, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((double**) &d_CAPE3km, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((double**) &d_ELP, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((double**) &d_ELT, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((double**) &d_Piter, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((double**) &d_Titer, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((double**) &d_Tparcel, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((double**) &d_prevTparcel, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((double**) &d_LFC, sizeof(double) * N));
	
	CUDA_CHECK(cudaMalloc((double**) &d_found, sizeof(unsigned char) * N));
	
	InitializeArray<unsigned char> (d_found, 0, N, stream);

	//CUDA_CHECK(cudaMemcpyAsync(d_TenvLCL, &TenvLCL[0], sizeof(double) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_Titer, &T[0], sizeof(double) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_prevTparcel, d_Titer, sizeof(double) * N, cudaMemcpyDeviceToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_Piter, &P[0], sizeof(double) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_LFC, &P[0], sizeof(double) * N, cudaMemcpyHostToDevice, stream));

	CUDA_CHECK(cudaMemcpyAsync(d_found, &found[0], sizeof(unsigned char) * N, cudaMemcpyHostToDevice, stream));
	
	//CUDA_CHECK(cudaMemcpyAsync(d_LCLP, d_Piter, sizeof(double) * N, cudaMemcpyDeviceToDevice, stream));
	
	InitializeArray<double> (d_CAPE, 0., N, stream);
	InitializeArray<double> (d_CAPE1040, 0., N, stream);
	InitializeArray<double> (d_CAPE3km, 0., N, stream);
	
	InitializeArray<double> (d_ELP, kFloatMissing, N, stream);
	InitializeArray<double> (d_ELT, kFloatMissing, N, stream);

	// For each grid point find the hybrid level that's below LFC and then pick the lowest level
	// among all grid points
		
	auto levels = h->LevelForHeight(myTargetInfo->Producer(), ::Max(P));
	
	level curLevel = levels.first;

	auto prevZenvInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("HL-M"), myTargetInfo->ForecastType());
	auto prevTenvInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType());
	auto prevPenvInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType());
	
	auto h_prevZenv = PrepareInfo(prevZenvInfo, stream);
	auto h_prevPenv = PrepareInfo(prevPenvInfo, stream);
	auto h_prevTenv = PrepareInfo(prevTenvInfo, stream);
	
	curLevel.Value(curLevel.Value() - 1);
	
	auto hPa100 = h->LevelForHeight(myTargetInfo->Producer(), 100.);

	while (curLevel.Value() > hPa100.first.Value())
	{
		auto PenvInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType());
		auto TenvInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType());
		auto ZenvInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("HL-M"), myTargetInfo->ForecastType());

		auto h_Zenv = PrepareInfo(ZenvInfo, stream);
		auto h_Penv = PrepareInfo(PenvInfo, stream);
		auto h_Tenv = PrepareInfo(TenvInfo, stream);
	
		MoistLiftKernel <<< gridSize, blockSize, 0, stream >>> (d_Titer, d_Piter, *h_Penv, d_Tparcel);
			
		CAPEKernel <<< gridSize, blockSize, 0, stream >>> (*h_Tenv, *h_Penv, *h_Zenv, *h_prevTenv, *h_prevPenv, *h_prevZenv, d_Tparcel, d_prevTparcel, d_LFC, d_CAPE, d_CAPE1040, d_CAPE3km, d_ELT, d_ELP, d_found, curLevel.Value(), hPa100.first.Value());
		
		CUDA_CHECK(cudaFree(h_prevZenv->values));
		CUDA_CHECK(cudaFree(h_prevTenv->values));
		CUDA_CHECK(cudaFree(h_prevPenv->values));
		
		CUDA_CHECK(cudaMemcpyAsync(d_prevTparcel, d_Tparcel, sizeof(double) * N, cudaMemcpyDeviceToDevice, stream));
		
		delete h_prevZenv;
		delete h_prevPenv;
		delete h_prevTenv;
		
		h_prevZenv = h_Zenv;
		h_prevTenv = h_Tenv;
		h_prevPenv = h_Penv;

		CopyLFCIteratorValuesKernel <<< gridSize, blockSize, 0, stream >>> (d_Titer, d_Tparcel, d_Piter, *h_Penv);
		
		curLevel.Value(curLevel.Value() - 1);

	}
	
	CUDA_CHECK(cudaFree(h_prevZenv->values));
	CUDA_CHECK(cudaFree(h_prevTenv->values));
	CUDA_CHECK(cudaFree(h_prevPenv->values));

	delete h_prevZenv;
	delete h_prevPenv;
	delete h_prevTenv;

#if 0
		
	// If the CAPE area is continued all the way to level 60 and beyond, we don't have an EL for that
	// (since integration is forcefully stopped)
	// In this case level 60 = EL
	
	for (size_t i = 0; i < CAPE.size(); i++)
	{
		if (CAPE[i] > 0 && ELT[i] == kFloatMissing)
		{
			TenvInfo->LocationIndex(i);
			PenvInfo->LocationIndex(i);
			
			ELT[i] = TenvInfo->Value();
			ELP[i] = PenvInfo->Value();
		}
	}
#endif
	
	std::vector<double> CAPE(T.size(), 0);
	std::vector<double> CAPE1040(T.size(), 0);
	std::vector<double> CAPE3km(T.size(), 0);
	std::vector<double> ELT(T.size(), kFloatMissing);
	std::vector<double> ELP(T.size(), kFloatMissing);

	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaMemcpyAsync(&CAPE[0], d_CAPE, sizeof(double) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(&CAPE1040[0], d_CAPE1040, sizeof(double) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(&CAPE3km[0], d_CAPE3km, sizeof(double) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(&ELT[0], d_ELT, sizeof(double) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(&ELP[0], d_ELP, sizeof(double) * N, cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaFree(d_Tparcel));
	CUDA_CHECK(cudaFree(d_prevTparcel));
	CUDA_CHECK(cudaFree(d_LFC));
	CUDA_CHECK(cudaFree(d_found));
	
	CUDA_CHECK(cudaStreamSynchronize(stream));
	
	CUDA_CHECK(cudaFree(d_CAPE));
	CUDA_CHECK(cudaFree(d_CAPE1040));
	CUDA_CHECK(cudaFree(d_CAPE3km));
	CUDA_CHECK(cudaFree(d_ELT));
	CUDA_CHECK(cudaFree(d_ELP));
	
	myTargetInfo->Param(ELTParam);
	myTargetInfo->Data().Set(ELT);
	
	myTargetInfo->Param(ELPParam);
	myTargetInfo->Data().Set(ELP);

	myTargetInfo->Param(CAPEParam);
	myTargetInfo->Data().Set(CAPE);
	
	myTargetInfo->Param(CAPE1040Param);
	myTargetInfo->Data().Set(CAPE1040);

	myTargetInfo->Param(CAPE3kmParam);
	myTargetInfo->Data().Set(CAPE3km);

}