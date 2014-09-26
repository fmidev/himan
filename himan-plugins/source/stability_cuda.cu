// System includes
#include <iostream>
#include <string>

// CUDA runtime
#include <cuda_runtime.h>

#include "stability_cuda.h"
#include "cuda_helper.h"
#include "metutil.h"

__global__ void himan::plugin::stability_cuda::Calculate(
		cdarr_t d_t850, cdarr_t d_t700, cdarr_t d_t500, cdarr_t d_td850, cdarr_t d_td700, // input for simple parameters
		cdarr_t d_t500m, cdarr_t d_td500m, cdarr_t d_p500m,  // input for more advanced traditional stability parameters
		cdarr_t d_u01, cdarr_t d_v01, cdarr_t d_u06, cdarr_t d_v06, // input for bulk shear
		darr_t d_ki, darr_t d_vti, darr_t d_cti, darr_t d_tti, darr_t d_si, darr_t d_li, // output for traditional stability
		darr_t d_bs01, darr_t d_bs06, // output for bulk shear
		options opts, int* d_missing)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{

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

			d_ki[idx] = himan::metutil::KI_(T850, T700, T500, TD850, TD700) - himan::constants::kKelvin;
			d_cti[idx] = himan::metutil::CTI_(TD850, T500);
			d_vti[idx] = himan::metutil::VTI_(T850, T500);
			d_tti[idx] = himan::metutil::TTI_(T850, T500, TD850); // CTI + VTI

			if (opts.li)
			{
				d_li[idx] = himan::metutil::LI_(T500, d_t500m[idx], d_td500m[idx], d_p500m[idx]);
				d_si[idx] = himan::metutil::SI_(T850, T500, TD850);
			}

			if (opts.bs01)
			{
				d_bs01[idx] = himan::metutil::BulkShear_(d_u01[idx], d_v01[idx]);
				d_bs06[idx] = himan::metutil::BulkShear_(d_u06[idx], d_v06[idx]);
			}
		}
	}
}

void himan::plugin::stability_cuda::Process(options& opts)
{

	cudaStream_t stream;

	CUDA_CHECK(cudaStreamCreate(&stream));

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
	double* d_bs01 = 0;
	double* d_bs06 = 0;
	double* d_t500m = 0;
	double* d_td500m = 0;
	double* d_p500m = 0;
	double* d_u01 = 0;
	double* d_v01 = 0;
	double* d_u06 = 0;
	double* d_v06 = 0;

	int* d_missing = 0;
	
	// Allocate memory on device

	size_t memsize = opts.N*sizeof(double);

	CUDA_CHECK(cudaMalloc((void **) &d_missing, sizeof(int)));

	Prepare(opts.t500, d_t500, memsize, stream);
	Prepare(opts.t700, d_t700, memsize, stream);
	Prepare(opts.t850, d_t850, memsize, stream);
	Prepare(opts.td700, d_td700, memsize, stream);
	Prepare(opts.td850, d_td850, memsize, stream);

	assert(d_t500);
	assert(d_t700);
	assert(d_t850);
	assert(d_td700);
	assert(d_td850);
	
	if (opts.li)
	{
		Prepare(opts.t500m, d_t500m, memsize, stream);
		Prepare(opts.td500m, d_td500m, memsize, stream);
		Prepare(opts.p500m, d_p500m, memsize, stream);

		CUDA_CHECK(cudaMalloc((void**) &d_li, memsize));

		assert(d_t500m);
		assert(d_td500m);
		assert(d_p500m);
	}

	if (opts.bs01)
	{
		Prepare(opts.u01, d_u01, memsize, stream);
		Prepare(opts.td500m, d_td500m, memsize, stream);
		Prepare(opts.p500m, d_p500m, memsize, stream);

		CUDA_CHECK(cudaMalloc((void**) &d_li, memsize));
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

	Calculate <<< gridSize, blockSize, 0, stream >>> (d_t850, d_t700, d_t500, d_td850, d_td700, d_t500m, d_td500m, d_p500m, d_u01, d_v01, d_u06, d_v06, d_ki, d_vti, d_cti, d_tti, d_si, d_li, d_bs01, d_bs06, opts, d_missing);

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

void himan::plugin::stability_cuda::Prepare(info_simple* source, double* devptr, size_t memsize, cudaStream_t& stream)
{
	CUDA_CHECK(cudaMalloc((void **) &devptr, memsize));
	
	if (source->packed_values)
	{
		// Unpack data and copy it back to host, we need it because its put back to cache
		source->packed_values->Unpack(devptr, source->size_x * source->size_y, &stream);
		CUDA_CHECK(cudaMemcpyAsync(source->values, &devptr, memsize, cudaMemcpyDeviceToHost, stream));
	}
	else
	{	
		CUDA_CHECK(cudaMemcpyAsync(devptr, source->values, memsize, cudaMemcpyHostToDevice, stream));
	}
}

void himan::plugin::stability_cuda::Prepare(const double* source, double* devptr, size_t memsize, cudaStream_t& stream)
{
	assert(source);
	CUDA_CHECK(cudaMalloc((void **) &devptr, memsize));
	CUDA_CHECK(cudaMemcpyAsync((void*) devptr, (const void*)source, memsize, cudaMemcpyHostToDevice, stream));
}
