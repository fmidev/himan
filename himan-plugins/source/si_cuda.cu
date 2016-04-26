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

using namespace himan;
using namespace himan::plugin;

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

	// 2. Unpack if needed, leave data to device and simultaneously copy it back to cpu (cache)
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

		auto c = GET_PLUGIN(cache);

		CUDA_CHECK(cudaStreamSynchronize(stream));

		tempGrid->PackedData().Clear();
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

void PrepareInfo(std::shared_ptr<himan::info> fullInfo, info_simple** h_info, cudaStream_t& stream)
{
	size_t N = (**h_info).size_x * (**h_info).size_y;
	assert(N > 0);

	// 1. Reserve memory at device for unpacked data
	double* d_arr = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<double**> (&d_arr), N * sizeof(double)));
	//CUDA_CHECK(cudaMalloc((double**) (&(*h_info)->values), N * sizeof(double)));
	(*h_info)->values = d_arr;

	// 2. Unpack if needed, leave data to device and simultaneously copy it back to cpu (cache)
	auto tempGrid = dynamic_cast<himan::regular_grid*> (fullInfo->Grid());

	if (tempGrid->IsPackedData())
	{
		assert(tempGrid->PackedData().ClassName() == "simple_packed" || tempGrid->PackedData().ClassName() == "jpeg_packed");
		assert(N > 0);
		assert(tempGrid->Data().Size() == N);

		double* arr = const_cast<double*> (tempGrid->Data().ValuesAsPOD());
		CUDA_CHECK(cudaHostRegister(reinterpret_cast<void*> (arr), sizeof(double) * N, 0));

		assert(arr);

		tempGrid->PackedData().Unpack((*h_info)->values, N, &stream);

		CUDA_CHECK(cudaMemcpyAsync(arr, (*h_info)->values, sizeof(double) * N, cudaMemcpyDeviceToHost, stream));

		auto c = GET_PLUGIN(cache);

		CUDA_CHECK(cudaStreamSynchronize(stream));

		tempGrid->PackedData().Clear();
		c->Insert(*fullInfo);	

		CUDA_CHECK(cudaHostUnregister(arr));

		(**h_info).packed_values = 0;

	}
	else
	{
		CUDA_CHECK(cudaMemcpyAsync((*h_info)->values, fullInfo->Data().ValuesAsPOD(), sizeof(double) * N, cudaMemcpyHostToDevice, stream));
	}

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
void MoistLiftKernel(const double* __restrict__ d_T, const double* __restrict__  d_P, info_simple d_Ptarget, double* __restrict__ d_Tparcel)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	assert(d_T);
	assert(d_P);

	if (idx < d_Ptarget.size_x * d_Ptarget.size_y)
	{
		assert(d_P[idx] > 10);
		assert(d_P[idx] < 1500);

		assert(d_Ptarget.values[idx] > 10);
		assert(d_Ptarget.values[idx] < 1500);

		assert(d_T[idx] > 100);
		assert(d_T[idx] < 350 || d_T[idx] == kFloatMissing);

		double T = metutil::MoistLift_(d_P[idx]*100, d_T[idx], d_Ptarget.values[idx]*100);

		assert(T > 100);
		assert(T < 350 || T == kFloatMissing);

		d_Tparcel[idx] = T;
	}
}

__global__
void LFCKernel(info_simple d_T, info_simple d_P, info_simple d_prevT, info_simple d_prevP, double* __restrict__ d_Tparcel, double* __restrict__ d_LCLP, double* __restrict__ d_LFCT, double* __restrict__ d_LFCP, unsigned char* __restrict__ d_found, int d_curLevel)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	assert(d_T.values);
	assert(d_P.values);

	if (idx < d_T.size_x * d_T.size_y)
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
		
		if (Tparcel != kFloatMissing && d_curLevel < 95 && (Tenv - Tparcel) > 30.)
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
void ThetaEKernel(info_simple d_T, info_simple d_RH, info_simple d_P, double* __restrict__ d_maxThetaE, double* __restrict__ d_Tresult, double* __restrict__ d_TDresult, unsigned char* __restrict__ d_found)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	assert(d_T.values);
	assert(d_RH.values);
	assert(d_P.values);
	
	if (idx < d_T.size_x * d_T.size_y)
	{
		double T = d_T.values[idx];
		double P = d_P.values[idx];
		double RH = d_RH.values[idx];
		double TD = metutil::DewPointFromRH_(T, RH);
		
		double& refThetaE = d_maxThetaE[idx];
		double ThetaE = metutil::ThetaE_(T, TD, P*100);

		if (P == kFloatMissing || P < 600.)
		{
			d_found[idx] = 1;
		}
		else
		{
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
	himan::level curLevel(kHybrid, 137);
	
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
	
	while (curLevel.Value() > 90)
	{
		auto TInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType());
		auto RHInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("RH-PRCNT"), myTargetInfo->ForecastType());
		auto PInfo = Fetch(conf, myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType());

		assert(TInfo && RHInfo && PInfo);
		assert(TInfo->Data().MissingCount() == 0);

		auto h_T = TInfo->ToSimple();
		PrepareInfo(TInfo, &h_T, stream);
		
		auto h_P = PInfo->ToSimple();
		PrepareInfo(PInfo, &h_P, stream);
		
		auto h_RH = RHInfo->ToSimple();
		PrepareInfo(RHInfo, &h_RH, stream);

		assert(h_T->values);
		assert(h_RH->values);
		assert(h_P->values);

		ThetaEKernel <<< gridSize, blockSize, 0, stream >>> (*h_T, *h_RH, *h_P, d_maxThetaE, d_Tresult, d_TDresult, d_found);

		std::vector<unsigned char> found(N, 0);
		CUDA_CHECK(cudaMemcpyAsync(&found[0], d_found, sizeof(unsigned char) * N, cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaStreamSynchronize(stream));

		CUDA_CHECK(cudaFree(h_P->values));
		CUDA_CHECK(cudaFree(h_RH->values));
		CUDA_CHECK(cudaFree(h_T->values));
		
		delete h_P;
		delete h_T;
		delete h_RH;

		curLevel.Value(curLevel.Value()-1);

		size_t foundCount = std::count(found.begin(), found.end(), 1);

		if (foundCount == found.size()) break;
	}
	
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
	//auto h = GET_PLUGIN(hitool);

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

	//auto levels = h->LevelForHeight(myTargetInfo->Producer(), CAPE::Max(P));

	//level curLevel = levels.first;
	level curLevel = level(kHybrid,137);
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
			found[i] = true;
			LFCT[i] = T[i];
			LFCP[i] = P[i];
			//Piter[i] = kFloatMissing;
		}
	}

	CUDA_CHECK(cudaMemcpyAsync(d_found, &found[0], sizeof(unsigned char) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_LFCT, &LFCT[0], sizeof(unsigned char) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_LFCP, &LFCP[0], sizeof(unsigned char) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaStreamSynchronize(stream));

	while (curLevel.Value() > 70)
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

		LFCKernel <<< gridSize, blockSize, 0, stream >>> (*h_Tenv, *h_Penv, *h_prevTenv, *h_prevPenv, d_Tparcel, d_LCLP, d_LFCT, d_LFCP, d_found, curLevel.Value());

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
