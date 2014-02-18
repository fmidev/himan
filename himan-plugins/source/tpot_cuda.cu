// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "tpot_cuda.h"

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
		if (opts.theta)
		{
			Theta(d_t, d_p, d_tp, opts, d_missing, idx);
		}
		if (opts.thetaw)
		{
			ThetaW(d_t, d_p, d_td, d_tpw, opts, d_missing, idx);
		}

	}
}
__device__ void himan::plugin::tpot_cuda::Theta(const double* __restrict__ d_t,
													const double* __restrict__ d_p,
													double* __restrict__ d_tp,
													options opts,
													int* d_missing, int idx)
{

	double P = (opts.is_constant_pressure) ? opts.p_const : d_p[idx];

	if (d_t[idx] == kFloatMissing || P == kFloatMissing)
	{
		atomicAdd(d_missing, 1);
		d_tp[idx] = kFloatMissing;
	}
	else
	{
		d_tp[idx] = (opts.t_base + d_t[idx]) * pow((1000 / (0.01 * P * opts.p_scale)), 0.28586);
	}

}

__device__ void himan::plugin::tpot_cuda::ThetaW(const double* __restrict__ d_t,
														const double* __restrict__ d_p,
														const double* __restrict__ d_td,
														double* __restrict__ d_tpw,
														options opts,
														int* d_missing, int idx)
{

   const double Pstep = 500; // Pa
   double value = kFloatMissing;

   // Search LCL level

   double Tint = 0, Pint = 0;

   LCL(d_p[idx], d_t[idx], d_td[idx], &Pint, &Tint);

   /*vector<double> LCL = util::LCL(P, T, TD);

   double Pint = LCL[0]; // Pa
   double Tint = LCL[1]; // C
*/
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
		   double Q = 0;
		   Gammas(Pint * 0.01, Tint - 273.15, &Q);

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

   d_tpw[idx] = value;
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

	double E0 = 0;
	Es(TD, &E0);

	//double Q = 0.622 * E0 / P;
	double C = (T+273.15) / pow(E0, kRCp);

	double TLCL = kFloatMissing;
	double PLCL = kFloatMissing;

	double Torig = T;
	double Porig = P;

	short nq = 0;

	while (++nq < 100)
	{
		double Qa = 0;
		Es(T, &Qa);
		double TEs = C * pow(Qa, kRCp);

		if (fabs(TEs - (T+273.15)) < 0.05)
		{
			TLCL = T;
			PLCL = pow(((TLCL+273.15)/(Torig+273.15)), (1/kRCp)) * P;

			*Pout = PLCL;
			*Tout = (TLCL == kFloatMissing) ? kFloatMissing : TLCL - 273.15; // C
			//ret[2] = Q;

			break;
		}
		else
		{
			Tstep = min((TEs - T - 273.15) / (2 * (nq+1)), 15.);
			T -= Tstep;
		}
	}

	// Fallback to slower method

	T = Torig;
	Tstep = 0.1;

	nq = 0;

	while (++nq <= 500)
	{
		double temp = 0;
		Es(T, &temp);

		if ((C * pow(temp, kRCp)-(T+273.15)) > 0)
		{
			T -= Tstep;
		}
		else
		{
			TLCL = T;
			PLCL = pow((TLCL + 273.15) / (Torig+273.15), (1/kRCp)) * Porig;

			*Pout = PLCL; // HPa
			*Tout = (TLCL == kFloatMissing) ? kFloatMissing : TLCL - 273.15; // C
			//ret[2] = Q;

			break;
		}
	}
}

__device__ void himan::plugin::tpot_cuda::Es(double T, double *out)
{
	
	if (T > -5)
	{
		*out = 6.107 * pow(10., (7.5*T/(237.0+T)));
	}
	else
	{
		*out = 6.107 * pow(10., (9.5*T/(265.5+T)));
	}

}

__device__ void himan::plugin::tpot_cuda::Gammas(double P, double T, double* out)
{
	// http://en.wikipedia.org/wiki/Lapse_rate#Saturated_adiabatic_lapse_rate ?
	// http://glossary.ametsoc.org/wiki/Pseudoadiabatic_lapse_rate

	double Q = 0;
	Es(T, &Q);

	Q = (Q * 0.622) / P;

	// unit changes

	P *= 100; // hpa --> pa
	T += 273.15; // c --> k

	double A = 287 * T / 1003.5 / P * (1+2.5e6*Q/287/T);

	*out = A / (1 + 0.622 / 1003.5 * (pow(2.5e6, 2) / 287 * Q / pow(T,2)));
}