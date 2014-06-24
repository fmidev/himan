// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "stability_cuda.h"
#include "cuda_helper.h"
#include "metutil.h"

__global__ void himan::plugin::stability_cuda::Calculate(cdarr_t d_t850, cdarr_t d_t700, cdarr_t d_t500, cdarr_t d_td850, cdarr_t d_td700,
		cdarr_t d_t500m, cdarr_t d_td500m, cdarr_t d_p500m, darr_t d_ki, darr_t d_vti, darr_t d_cti, darr_t d_tti, darr_t d_si, darr_t d_li, options opts, int* d_missing)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{

		d_ki[idx] = kFloatMissing;

		double T850 = d_t850[idx];
		double T700 = d_t700[idx];
		double T500 = d_t500[idx];
		double TD850 = d_td850[idx];
		double TD700 = d_td700[idx];
		
		if (T850 == kFloatMissing || T700 == kFloatMissing || T500 == kFloatMissing || TD850 == kFloatMissing || TD700 == kFloatMissing)
		{
			atomicAdd(d_missing, 1);
		}
		else
		{

			d_ki[idx] = (T850 - T500 + TD850 - (T700 - TD700)) - himan::constants::kKelvin;
			d_cti[idx] = TD850 - T500;
			d_vti[idx] = T850 - T500;
			d_tti[idx] = (TD850 - T500) + (T850 - T500); // CTI + VTI
			//d_si[idx] = SI(T850, T500, TD850, d_missing);

			if (opts.li)
			{
				//d_li[idx] = LI(T500, d_t500m[idx], d_td500m[idx], d_p500m[idx], d_missing);
			}
		}
	}
}

__device__
double himan::plugin::stability_cuda::LI(double T500, double T500m, double TD500m, double P500m, int* d_missing)
{
	lcl_t LCL = metutil::LCL_(50000, T500m, TD500m);

	double li = kFloatMissing;

	const double TARGET_PRESSURE = 50000;

	if (LCL.P == kFloatMissing)
	{
		return li;
	}

	if (LCL.P <= 85000)
	{
		// LCL pressure is below wanted pressure, no need to do wet-adiabatic
		// lifting

		double dryT = metutil::DryLift_(P500m, T500m, TARGET_PRESSURE);

		if (dryT != kFloatMissing)
		{
			li = T500 - dryT;
		}
	}
	else
	{
		// Grid point is inside or above cloud

		double wetT = metutil::MoistLift_(P500m, T500m, TD500m, TARGET_PRESSURE);

		if (wetT != kFloatMissing)
		{
			li = T500 - wetT;
		}
	}

	return li;
}

__device__
double himan::plugin::stability_cuda::SI(double T850, double T500, double TD850, int* d_missing)
{
				
	lcl_t LCL = himan::metutil::LCL_(85000., T850, TD850);

	double si = kFloatMissing;
	
	if (LCL.P == kFloatMissing)
	{
		atomicAdd(d_missing, 1);
	}
	else if (LCL.P <= 85000.)
	{
		// LCL pressure is below wanted pressure, no need to do wet-adiabatic
		// lifting

		double dryT = himan::metutil::DryLift_(85000., T850, 50000.);

		if (dryT != kFloatMissing)
		{
			si = T500 - dryT;
		}
	}
	else
	{
		// Grid point is inside or above cloud

		double wetT = himan::metutil::MoistLift_(85000., T850, TD850, 50000.);

		if (wetT != kFloatMissing)
		{
			si = T500 - wetT;
		}
	}

	return si;
}

void himan::plugin::stability_cuda::Process(options& opts)
{

	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

	// Allocate device arrays

	double* d_t500 = 0;
	double* d_t700 = 0;
	double* d_t850 = 0;
	double* d_td700 = 0;
	double* d_td850 = 0;
	double* d_ki = 0;
	double* d_si = 0;
	double* d_li = 0;
	double* d_vti = 0;
	double* d_cti = 0;
	double* d_tti = 0;
	double* d_t500m = 0;
	double* d_td500m = 0;
	double* d_p500m = 0;
	
	int* d_missing = 0;

	// Allocate memory on device

	size_t memsize = opts.N*sizeof(double);

	CUDA_CHECK(cudaMalloc((void **) &d_missing, sizeof(int)));

	Prepare(opts.t500, &d_t500, opts.N, stream);
	Prepare(opts.t700, &d_t700, opts.N, stream);
	Prepare(opts.t850, &d_t850, opts.N, stream);
	Prepare(opts.td700, &d_td700, opts.N, stream);
	Prepare(opts.td850, &d_td850, opts.N, stream);

	assert(d_t500);
	assert(d_t700);
	assert(d_t850);
	assert(d_td700);
	assert(d_td850);
	
	if (opts.li)
	{
		Prepare(opts.t500m, d_t500m, opts.N, stream);
		Prepare(opts.td500m, d_td500m, opts.N, stream);
		Prepare(opts.p500m, d_p500m, opts.N, stream);

		CUDA_CHECK(cudaMalloc((void**) &d_li, memsize));

		assert(d_t500m);
		assert(d_td500m);
		assert(d_p500m);
	}

	int src=0;

	CUDA_CHECK(cudaMemcpyAsync(d_missing, &src, sizeof(int), cudaMemcpyHostToDevice, stream));

	// dims

	const int blockSize = 256;
	const int gridSize = opts.N/blockSize + (opts.N%blockSize == 0?0:1);

	CUDA_CHECK(cudaMalloc((void**) &d_ki, memsize));
	CUDA_CHECK(cudaMalloc((void**) &d_vti, memsize));
	CUDA_CHECK(cudaMalloc((void**) &d_cti, memsize));
	CUDA_CHECK(cudaMalloc((void**) &d_tti, memsize));
	CUDA_CHECK(cudaMalloc((void**) &d_si, memsize));

	CUDA_CHECK(cudaStreamSynchronize(stream));

	Calculate <<< gridSize, blockSize, 0, stream >>> (d_t850, d_t700, d_t500, d_td850, d_td700, d_t500m, d_td500m, d_p500m, d_ki, d_vti, d_cti, d_tti, d_si, d_li, opts, d_missing);

	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaMemcpyAsync(opts.ki->values, d_ki, memsize, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(opts.si->values, d_si, memsize, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(opts.cti->values, d_cti, memsize, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(opts.vti->values, d_vti, memsize, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(opts.tti->values, d_tti, memsize, cudaMemcpyDeviceToHost, stream));

	if (opts.li)
	{
		CUDA_CHECK(cudaMemcpyAsync(opts.li->values, d_li, memsize, cudaMemcpyDeviceToHost, stream));
	}

	CUDA_CHECK(cudaMemcpyAsync(&opts.missing, d_missing, sizeof(int), cudaMemcpyDeviceToHost, stream));
	
	// block until the stream has completed
	CUDA_CHECK(cudaStreamSynchronize(stream));

	// Free device memory

	CUDA_CHECK(cudaFree(d_t850));
	CUDA_CHECK(cudaFree(d_t700));
	CUDA_CHECK(cudaFree(d_t500));
	CUDA_CHECK(cudaFree(d_td850));
	CUDA_CHECK(cudaFree(d_td700));
	CUDA_CHECK(cudaFree(d_si));
	CUDA_CHECK(cudaFree(d_ki));
	CUDA_CHECK(cudaFree(d_vti));
	CUDA_CHECK(cudaFree(d_cti));
	CUDA_CHECK(cudaFree(d_tti));

	if (opts.li)
	{
		CUDA_CHECK(cudaFree(d_li));
	}
	
	CUDA_CHECK(cudaFree(d_missing));

	CUDA_CHECK(cudaStreamDestroy(stream)); // this blocks
}

void himan::plugin::stability_cuda::Prepare(info_simple* source, double** devptr, size_t memsize, cudaStream_t& stream)
{
	if (source->packed_values)
	{
		// Unpack data and copy it back to host, we need it because its put back to cache
		*devptr = source->packed_values->Unpack(&stream);
		CUDA_CHECK(cudaMemcpyAsync(source->values, *devptr, memsize, cudaMemcpyDeviceToHost, stream));
	}
	else
	{
		CUDA_CHECK(cudaMalloc((void **) &devptr, memsize));
		CUDA_CHECK(cudaMemcpyAsync(devptr, source->values, memsize, cudaMemcpyHostToDevice, stream));
	}
}

void himan::plugin::stability_cuda::Prepare(const double* source, double* devptr, size_t memsize, cudaStream_t& stream)
{
	CUDA_CHECK(cudaMalloc((void **) &devptr, memsize));
	CUDA_CHECK(cudaMemcpyAsync(devptr, source, memsize, cudaMemcpyHostToDevice, stream));
}