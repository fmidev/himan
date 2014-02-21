// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "tpot_cuda.h"
//#include "stdio.h"

__global__ void himan::plugin::tpot_cuda::Calculate(const double* __restrict__ d_t,
													const double* __restrict__ d_p,
													const double* __restrict__ d_td,
													double* __restrict__ d_tp,
													double* __restrict__ d_tpw,
													double* __restrict__ d_tpe,
													options opts,
													int* d_missing)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		double P = (opts.is_constant_pressure) ? opts.p_const : d_p[idx];

		if (opts.theta)
		{
			d_tp[idx] = Theta(d_t[idx], P, opts, d_missing);
		}
		if (opts.thetaw)
		{
			d_tpw[idx] = ThetaW(d_t[idx], P, d_td[idx], opts, d_missing);
		}
		if (opts.thetae)
		{
			d_tpe[idx] = ThetaE(d_t[idx], P, d_td[idx], opts, d_missing);
		}

	}
}
__device__ double himan::plugin::tpot_cuda::Theta(double T, double P, options opts,int* d_missing)
{

	double theta;
	
	if (T == kFloatMissing || P == kFloatMissing)
	{
		if (d_missing)
		{
			atomicAdd(d_missing, 1);
		}
		
		theta = kFloatMissing;
	}
	else
	{
		theta = (opts.t_base + T) * pow((1000 / (0.01 * P * opts.p_scale)), 0.28586);
	}

	return theta;

}

__device__ double himan::plugin::tpot_cuda::ThetaW(double T, double P, double TD, options opts, int* d_missing)
{
	
	const double Pstep = 500; // Pa
	double value = kFloatMissing;

	// Search LCL level

	double Tint = 0, Pint = 0;

	LCL(P, T, TD, &Pint, &Tint);

	int i = 0;

	if (Tint != kFloatMissing && Pint != kFloatMissing)
	{

		/*
		* Units: Temperature in Kelvins, Pressure in Pascals
		*/

		Tint += 273.15;
		Pint *= 100;

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
			double Q = Gammas(Pint * 0.01, Tint - 273.15);

			Tint = T0 + Q * Z;

			if (i > 2)
			{
				T0 = TA;
			}

			Pint += Pstep;

			if (Pint >= 1e5)
			{
				value = Tint - 273.15;
				break;
			}
		}
	}

	return value;
}

__device__ double himan::plugin::tpot_cuda::ThetaE(double T, double P, double TD, options opts, int* d_missing)
{

	// Search LCL level

	double TLCL = 0, PLCL = 0;
	const double kEp = 0.622;
	const double kL = 2.5e6;
	const double kCp = 1003.5;
	
	LCL(P, T, TD, &PLCL, &TLCL);

	double theta = Theta(T, P, opts, 0) - 273.15;

	double ZEs = Es(TLCL) * 0.01;
	double ZQs = kEp * (ZEs / (P*0.01 - ZEs));

	return 273.15 + theta * exp(kL * ZQs / kCp / (TLCL));

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

	int* d_missing = 0;

	size_t memsize = opts.N * sizeof(double);

	// Allocate memory on device

	CUDA_CHECK(cudaMalloc((void **) &d_missing, sizeof(int)));

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

	if (opts.t->packed_values)
	{
		// Unpack data and copy it back to host, we need it because its put back to cache
		d_t = opts.t->packed_values->Unpack(&stream);
		CUDA_CHECK(cudaMemcpyAsync(opts.t->values, d_t, memsize, cudaMemcpyDeviceToHost, stream));
	}
	else
	{
		CUDA_CHECK(cudaMalloc((void **) &d_t, memsize));
		CUDA_CHECK(cudaMemcpyAsync(d_t, opts.t->values, memsize, cudaMemcpyHostToDevice, stream));
	}

	if (!opts.is_constant_pressure)
	{
		if (opts.p->packed_values)
		{
			d_p = opts.p->packed_values->Unpack(&stream);
			CUDA_CHECK(cudaMemcpyAsync(opts.p->values, d_p, memsize, cudaMemcpyDeviceToHost, stream));
		}
		else
		{
			CUDA_CHECK(cudaMalloc((void **) &d_p, memsize));
			CUDA_CHECK(cudaMemcpyAsync(d_p, opts.p->values, memsize, cudaMemcpyHostToDevice, stream));
		}
	}

	// td

	if (opts.thetaw || opts.thetae)
	{
		if (opts.td->packed_values)
		{
			// Unpack data and copy it back to host, we need it because its put back to cache
			d_td = opts.td->packed_values->Unpack(&stream);
			CUDA_CHECK(cudaMemcpyAsync(opts.td->values, d_td, memsize, cudaMemcpyDeviceToHost, stream));
		}
		else
		{
			CUDA_CHECK(cudaMalloc((void **) &d_td, memsize));
			CUDA_CHECK(cudaMemcpyAsync(d_td, opts.td->values, memsize, cudaMemcpyHostToDevice, stream));
		}
	}
	
	int src=0;

	CUDA_CHECK(cudaMemcpyAsync(d_missing, &src, sizeof(int), cudaMemcpyHostToDevice, stream));

	// dims

	const int blockSize = 512;
	const int gridSize = opts.N/blockSize + (opts.N%blockSize == 0?0:1);
		
	CUDA_CHECK(cudaStreamSynchronize(stream));
	
	Calculate <<< gridSize, blockSize, 0, stream >>> (d_t, d_p, d_td, d_tp, d_tpw, d_tpe, opts, d_missing);

	// block until the device has completed
	CUDA_CHECK(cudaStreamSynchronize(stream));

	// check if kernel execution generated an error

	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	CUDA_CHECK(cudaStreamSynchronize(stream));

	// Retrieve result from device
	CUDA_CHECK(cudaMemcpyAsync(&opts.missing, d_missing, sizeof(int), cudaMemcpyDeviceToHost, stream));

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


	CUDA_CHECK(cudaFree(d_missing));
	CUDA_CHECK(cudaFree(d_t));
	
	if (d_p)
	{
		CUDA_CHECK(cudaFree(d_p));
	}

	CUDA_CHECK(cudaStreamDestroy(stream));
}

__device__ void himan::plugin::tpot_cuda::LCL(double P, double T, double TD, double* Pout, double* Tout)
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

			*Pout = PLCL * 100; // Pa
			*Tout = (TLCL == kFloatMissing) ? kFloatMissing : TLCL; // C
			//ret[2] = Q;
	
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
	//printf ("SLOW LCL\n");
	while (++nq <= 100)
	{
		//printf("%f\n", C * pow(Qa*0.01, kRCp) - T );
		if (C * pow(Es(T)*0.01, kRCp) - T > 0)
		{
			T -= Tstep;
		}
		else
		{
			TLCL = T;
			PLCL = pow((TLCL / Torig), (1/kRCp)) * Porig;

			*Pout = PLCL * 100; // Pa
			*Tout = (TLCL == kFloatMissing) ? kFloatMissing : TLCL; // C
			//ret[2] = Q;

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

	double A = kRd * T / kCp / P * (1+2.5e6*Q/kRd/T);

	return A / (1 + kEp / kCp * ((kL*kL) / kRd * Q / (T*T)));
}