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
			d_tp[idx] = Theta(opts.t_base + d_t[idx], P * opts.p_scale, opts);
		}
		if (opts.thetaw)
		{
			d_tpw[idx] = ThetaW(opts.t_base + d_t[idx], opts.p_scale * P, opts.td_base + d_td[idx], opts);
		}
		if (opts.thetae)
		{
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

	// Allocate memory on device

	if (opts.theta)
	{
		CUDA_CHECK(cudaMalloc((void **) &d_tp, memsize));
	}

	if (opts.thetaw)
	{
		CUDA_CHECK(cudaMalloc((void **) &d_tpw, memsize));
	}

	if (opts.thetae)
	{
		CUDA_CHECK(cudaMalloc((void **) &d_tpe, memsize));

		if (!opts.theta)
		{
			CUDA_CHECK(cudaMalloc((void **) &d_tp, memsize));
		}
	}

	CUDA_CHECK(cudaMalloc((void **) &d_t, memsize));
	
	if (opts.t->packed_values)
	{
		// Unpack data and copy it back to host, we need it because its put back to cache
		opts.t->packed_values->Unpack(d_t, opts.N, &stream);
		CUDA_CHECK(cudaMemcpyAsync(opts.t->values, d_t, memsize, cudaMemcpyDeviceToHost, stream));
	}
	else
	{
		CUDA_CHECK(cudaMemcpyAsync(d_t, opts.t->values, memsize, cudaMemcpyHostToDevice, stream));
	}

	if (!opts.is_constant_pressure)
	{
		CUDA_CHECK(cudaMalloc((void **) &d_p, memsize));

		if (opts.p->packed_values)
		{
			opts.p->packed_values->Unpack(d_p, opts.N, &stream);
			CUDA_CHECK(cudaMemcpyAsync(opts.p->values, d_p, memsize, cudaMemcpyDeviceToHost, stream));
		}
		else
		{
			CUDA_CHECK(cudaMemcpyAsync(d_p, opts.p->values, memsize, cudaMemcpyHostToDevice, stream));
		}
	}

	// td

	if (opts.thetaw || opts.thetae)
	{
		CUDA_CHECK(cudaMalloc((void **) &d_td, memsize));

		if (opts.td->packed_values)
		{
			// Unpack data and copy it back to host, we need it because its put back to cache
			opts.td->packed_values->Unpack(d_td, opts.N, &stream);
			CUDA_CHECK(cudaMemcpyAsync(opts.td->values, d_td, memsize, cudaMemcpyDeviceToHost, stream));
		}
		else
		{
			CUDA_CHECK(cudaMemcpyAsync(d_td, opts.td->values, memsize, cudaMemcpyHostToDevice, stream));
		}
	}
	
	// dims

	const int blockSize = 512;
	const int gridSize = opts.N/blockSize + (opts.N%blockSize == 0?0:1);
		
	CUDA_CHECK(cudaStreamSynchronize(stream));
	
	Calculate <<< gridSize, blockSize, 0, stream >>> (d_t, d_p, d_td, d_tp, d_tpw, d_tpe, opts);

	// block until the device has completed
	CUDA_CHECK(cudaStreamSynchronize(stream));

	// check if kernel execution generated an error

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	CUDA_CHECK(cudaStreamSynchronize(stream));

	// Retrieve result from device

	if (opts.theta)
	{
		CUDA_CHECK(cudaMemcpyAsync(opts.tp->values, d_tp, memsize, cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaFree(d_tp));
	}

	if (opts.thetaw)
	{
		CUDA_CHECK(cudaMemcpyAsync(opts.tpw->values, d_tpw, memsize, cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaFree(d_tpw));
	}

	if (opts.thetae)
	{
		CUDA_CHECK(cudaMemcpyAsync(opts.tpe->values, d_tpe, memsize, cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaFree(d_tpe));
	}

	CUDA_CHECK(cudaFree(d_t));
	
	if (d_p)
	{
		CUDA_CHECK(cudaFree(d_p));
	}

	if (d_td)
	{
		CUDA_CHECK(cudaFree(d_td));
	}

	CUDA_CHECK(cudaStreamDestroy(stream));
}

#if 0
__device__ void himan::plugin::tpot_cuda::LCL(double P, double T, double TD, double& Pout, double& Tout)
{
	// starting T step

	double Tstep = 0.05;

	const double kRCp = 0.286;

	// saturated vapor pressure

	double E0 = Es(TD) * 0.01; // HPa

	P *= 0.01; // hPa
		
	double C = T / pow(E0, kRCp);

	double TLCL = kFloatMissing;
	double PLCL = kFloatMissing;

	double Torig = T;

	short nq = 0;

	while (++nq < 100)
	{

		double TEs = C * pow(Es(T) * 0.01, kRCp);

		if (fabs(TEs - T) < 0.05)
		{
			TLCL = T;
			PLCL = pow((TLCL / Torig), (1/kRCp)) * P;

			Pout = PLCL * 100; // Pa
			Tout = (TLCL == kFloatMissing) ? kFloatMissing : TLCL; // C

			return;
		}
		else
		{
			Tstep = min((TEs - T) / (2 * (nq+1)), 15.);
			T -= Tstep;
		}
	}

	// Fallback to slower method

	T = Torig;
	Tstep = 0.1;

	nq = 0;

	double Porig = P;

	while (++nq <= 500)
	{
		if (C * pow(Es(T)*0.01, kRCp) - T > 0)
		{
			T -= Tstep;
		}
		else
		{
			TLCL = T;
			PLCL = pow((TLCL / Torig), (1/kRCp)) * Porig;

			Pout = PLCL * 100; // Pa
			Tout = (TLCL == kFloatMissing) ? kFloatMissing : TLCL; // C

			break;
		}
	}


}

__device__ double himan::plugin::tpot_cuda::Es(double T)
{
	T -= 273.15;

	double Es;
	
	if (T > -5)
	{
		Es = 6.107 * pow(10., (7.5*T/(237.0+T)));
	}
	else
	{
		Es = 6.107 * pow(10., (9.5*T/(265.5+T)));
	}

	return 100 * Es;
}

__device__ double himan::plugin::tpot_cuda::Gammas(double P, double T)
{
	// http://en.wikipedia.org/wiki/Lapse_rate#Saturated_adiabatic_lapse_rate ?
	// http://glossary.ametsoc.org/wiki/Pseudoadiabatic_lapse_rate

	const double kEp = 0.622;
	const double kRd = 287;
	const double kL = 2.5e6;
	const double kCp = 1003.5;

	double Q = kEp * (Es(T) * 0.01) / (P * 0.01);

	double A = kRd * T / kCp / P * (1+kL*Q/kRd/T);

	return A / (1 + kEp / kCp * (kL*kL / kRd * Q / (T*T)));
}
#endif