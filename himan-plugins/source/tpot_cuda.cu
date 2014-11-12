// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>
#include "stdio.h"
#include "tpot_cuda.h"
#include "metutil.h"

__global__ void himan::plugin::tpot_cuda::Calculate(const double* __restrict__ d_t,
													const double* __restrict__ d_p,
													const double* __restrict__ d_td,
													double* __restrict__ d_tp,
													double* __restrict__ d_tpw,
													double* __restrict__ d_tpe,
													options opts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		double P = (opts.is_constant_pressure) ? opts.p_const : d_p[idx];

		if (opts.theta)
		{
			d_tp[idx] = kFloatMissing;
			d_tp[idx] = Theta(opts.t_base + d_t[idx], P * opts.p_scale, opts);
		}
		if (opts.thetaw)
		{
			d_tpw[idx] = kFloatMissing;
			d_tpw[idx] = ThetaW(opts.t_base + d_t[idx], opts.p_scale * P, opts.td_base + d_td[idx], opts);
		}
		if (opts.thetae)
		{
			d_tpe[idx] = kFloatMissing;
			d_tpe[idx] = ThetaE(opts.t_base + d_t[idx], opts.p_scale * P, opts.td_base + d_td[idx], opts);
		}

	}
}
__device__ double himan::plugin::tpot_cuda::Theta(double T, double P, options opts)
{

	double theta = kFloatMissing;
	
	if (T != kFloatMissing && P != kFloatMissing)
	{
		theta = T * pow((1000 / (0.01 * P)), 0.28586);
	}

	return theta;

}

__device__ double himan::plugin::tpot_cuda::ThetaW(double T, double P, double TD, options opts)
{

	double value = kFloatMissing;

	if (T != kFloatMissing && P != kFloatMissing && TD != kFloatMissing)
	{

		const double Pstep = 500; // Pa

		// Search LCL level

		lcl_t LCL = himan::metutil::LCL_(P, T, TD);

		double Tint = LCL.T;
		double Pint = LCL.P;

		int i = 0;

		if (Tint != kFloatMissing && Pint != kFloatMissing)
		{

			/*
			* Units: Temperature in Kelvins, Pressure in Pascals
			*/

			double T0 = Tint;

			double Z = kFloatMissing;

			while (++i < 500) // usually we don't reach this value
			{
				double TA = Tint;

				if (i <= 2)
				{
					Z = i * Pstep/2;
				}
				else
				{
					Z = 2 * Pstep;
				}

				// Gammas() takes hPa
				Tint = T0 + himan::metutil::Gammas_(Pint, Tint) * Z;

				if (i > 2)
				{
					T0 = TA;
				}

				Pint += Pstep;

				if (Pint >= 1e5)
				{
					value = Tint;
					break;
				}
			}
		}
	}

	return value;
}

__device__ double himan::plugin::tpot_cuda::ThetaE(double T, double P, double TD, options opts)
{

	double value = kFloatMissing;

	if (T != kFloatMissing && P != kFloatMissing & TD != kFloatMissing)
	{
		// Search LCL level

		const double kEp = 0.622;
		const double kL = 2.5e6;
		const double kCp = 1003.5;

		lcl_t LCL = himan::metutil::LCL_(P, T, TD);

		if (LCL.T != kFloatMissing)
		{
			double theta = Theta(T, P, opts) - himan::constants::kKelvin; // C

			// No need to check theta for kFloatMissing since Theta() always returns
			// value if T and P are != kFloatMissing
		
			double ZEs = himan::metutil::Es_(LCL.T) * 0.01;
			double ZQs = kEp * (ZEs / (P*0.01 - ZEs));

			value = 273.15 + theta * exp(kL * ZQs / kCp / (LCL.T));
		}
	}

	return value;
}

void himan::plugin::tpot_cuda::Process(options& opts)
{
	
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	double* d_t = 0;
	double* d_p = 0;
	double* d_td = 0;
	double* d_tp = 0;
	double* d_tpw = 0;
	double* d_tpe = 0;

	size_t memsize = opts.N * sizeof(double);

	// dims

	const int blockSize = 512;
	const int gridSize = opts.N/blockSize + (opts.N%blockSize == 0?0:1);
		
	// Allocate memory on device

	if (opts.theta || opts.thetae)
	{
		CUDA_CHECK(cudaMalloc((void **) &d_tp, memsize));
		PrepareInfo(opts.tp);
	}

	if (opts.thetaw)
	{
		CUDA_CHECK(cudaMalloc((void **) &d_tpw, memsize));
		PrepareInfo(opts.tpw);
	}

	if (opts.thetae)
	{
		CUDA_CHECK(cudaMalloc((void **) &d_tpe, memsize));
		PrepareInfo(opts.tpe);

	}

	CUDA_CHECK(cudaMalloc((void **) &d_t, memsize));

	PrepareInfo(opts.t, d_t, stream);

	if (!opts.is_constant_pressure)
	{
		CUDA_CHECK(cudaMalloc((void **) &d_p, memsize));

		PrepareInfo(opts.p, d_p, stream);

	}

	// td

	if (opts.thetaw || opts.thetae)
	{
		CUDA_CHECK(cudaMalloc((void **) &d_td, memsize));

		PrepareInfo(opts.td, d_td, stream);
	}

	CUDA_CHECK(cudaStreamSynchronize(stream));
	
	Calculate <<< gridSize, blockSize, 0, stream >>> (d_t, d_p, d_td, d_tp, d_tpw, d_tpe, opts);

	// block until the device has completed
	CUDA_CHECK(cudaStreamSynchronize(stream));

	// check if kernel execution generated an error

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	// Retrieve result from device

	if (opts.theta)
	{
		ReleaseInfo(opts.tp, d_tp, stream);
		CUDA_CHECK(cudaFree(d_tp));
	}

	if (opts.thetaw)
	{
		ReleaseInfo(opts.tpw, d_tpw, stream);
		CUDA_CHECK(cudaFree(d_tpw));
	}

	if (opts.thetae)
	{
		ReleaseInfo(opts.tpe, d_tpe, stream);
		CUDA_CHECK(cudaFree(d_tpe));
	}

	CUDA_CHECK(cudaFree(d_t));
	ReleaseInfo(opts.t);

	if (d_p)
	{
		CUDA_CHECK(cudaFree(d_p));
		ReleaseInfo(opts.p);
	}

	if (d_td)
	{
		CUDA_CHECK(cudaFree(d_td));
		ReleaseInfo(opts.td);
	}

	CUDA_CHECK(cudaStreamDestroy(stream));
}
