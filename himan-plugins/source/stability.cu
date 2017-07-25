// System includes
#include <iostream>
#include <string>

#include "cuda_plugin_helper.h"
#include "metutil.h"
#include "stability.cuh"

__global__ void himan::plugin::stability_cuda::Calculate(
    cdarr_t d_t850, cdarr_t d_t700, cdarr_t d_t500, cdarr_t d_td850, cdarr_t d_td700,  // input for simple parameters
    cdarr_t d_t500m, cdarr_t d_td500m, cdarr_t d_p500m,  // input for more advanced traditional stability parameters
    cdarr_t d_u01, cdarr_t d_v01, cdarr_t d_u06, cdarr_t d_v06,  // input for bulk shear
    darr_t d_ki, darr_t d_vti, darr_t d_cti, darr_t d_tti, darr_t d_si,
    darr_t d_li,                   // output for traditional stability
    darr_t d_bs01, darr_t d_bs06,  // output for bulk shear
    options opts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.N)
	{
		double T850 = d_t850[idx];
		double T700 = d_t700[idx];
		double T500 = d_t500[idx];
		double TD850 = d_td850[idx];
		double TD700 = d_td700[idx];

		d_ki[idx] = MissingDouble();
		d_cti[idx] = MissingDouble();
		d_vti[idx] = MissingDouble();
		d_tti[idx] = MissingDouble();

		d_ki[idx] = himan::metutil::KI_(T850, T700, T500, TD850, TD700);
		d_cti[idx] = himan::metutil::CTI_(T500, TD850);
		d_vti[idx] = himan::metutil::VTI_(T850, T500);
		d_tti[idx] = himan::metutil::TTI_(T850, T500, TD850);  // CTI + VTI

		if (opts.li)
		{
			d_li[idx] = himan::metutil::LI_(T500, d_t500m[idx], d_td500m[idx], d_p500m[idx]);
			d_si[idx] = himan::metutil::SI_(T850, T500, TD850);
		}

		if (opts.bs01)
		{
			double u01 = d_u01[idx];
			double v01 = d_v01[idx];
			double u06 = d_u06[idx];
			double v06 = d_v06[idx];

			d_bs01[idx] = himan::metutil::BulkShear_(u01, v01);
			d_bs06[idx] = himan::metutil::BulkShear_(u06, v06);
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

	// Allocate memory on device

	size_t memsize = opts.N * sizeof(double);

	CUDA_CHECK(cudaMalloc((void**)&d_ki, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_vti, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_cti, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_tti, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_t500, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_t700, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_t850, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_td700, memsize));
	CUDA_CHECK(cudaMalloc((void**)&d_td850, memsize));

	PrepareInfo(opts.t500, d_t500, stream);
	PrepareInfo(opts.t700, d_t700, stream);
	PrepareInfo(opts.t850, d_t850, stream);
	PrepareInfo(opts.td700, d_td700, stream);
	PrepareInfo(opts.td850, d_td850, stream);
	PrepareInfo(opts.ki);
	PrepareInfo(opts.tti);
	PrepareInfo(opts.cti);
	PrepareInfo(opts.vti);

	// dims

	const int blockSize = 512;
	const int gridSize = opts.N / blockSize + (opts.N % blockSize == 0 ? 0 : 1);

	assert(d_t500);
	assert(d_t700);
	assert(d_t850);
	assert(d_td700);
	assert(d_td850);

	if (opts.li)
	{
		CUDA_CHECK(cudaMalloc((void**)&d_si, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_li, memsize));

		PrepareInfo(opts.li);
		PrepareInfo(opts.si);

		Prepare(opts.t500m, &d_t500m, memsize, stream);
		Prepare(opts.td500m, &d_td500m, memsize, stream);
		Prepare(opts.p500m, &d_p500m, memsize, stream);

		assert(d_t500m);
		assert(d_td500m);
		assert(d_p500m);
	}

	if (opts.bs01)
	{
		CUDA_CHECK(cudaMalloc((void**)&d_bs01, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_bs06, memsize));

		PrepareInfo(opts.bs01);
		PrepareInfo(opts.bs06);

		Prepare(opts.u01, &d_u01, memsize, stream);
		Prepare(opts.v01, &d_v01, memsize, stream);
		Prepare(opts.u06, &d_u06, memsize, stream);
		Prepare(opts.v06, &d_v06, memsize, stream);
	}

	CUDA_CHECK(cudaStreamSynchronize(stream));

	Calculate<<<gridSize, blockSize, 0, stream>>>(d_t850, d_t700, d_t500, d_td850, d_td700, d_t500m, d_td500m, d_p500m,
	                                              d_u01, d_v01, d_u06, d_v06, d_ki, d_vti, d_cti, d_tti, d_si, d_li,
	                                              d_bs01, d_bs06, opts);

	CUDA_CHECK(cudaStreamSynchronize(stream));

	himan::ReleaseInfo(opts.ki, d_ki, stream);
	himan::ReleaseInfo(opts.cti, d_cti, stream);
	himan::ReleaseInfo(opts.tti, d_tti, stream);
	himan::ReleaseInfo(opts.vti, d_vti, stream);

	himan::ReleaseInfo(opts.t500);
	himan::ReleaseInfo(opts.t700);
	himan::ReleaseInfo(opts.t850);
	himan::ReleaseInfo(opts.td700);
	himan::ReleaseInfo(opts.td850);

	if (opts.li)
	{
		himan::ReleaseInfo(opts.li, d_li, stream);
		himan::ReleaseInfo(opts.si, d_si, stream);
		CUDA_CHECK(cudaFree(d_li));
		CUDA_CHECK(cudaFree(d_si));
	}

	if (opts.bs01)
	{
		himan::ReleaseInfo(opts.bs01, d_bs01, stream);
		himan::ReleaseInfo(opts.bs06, d_bs06, stream);

		CUDA_CHECK(cudaFree(d_bs01));
		CUDA_CHECK(cudaFree(d_bs06));
	}

	// block until the stream has completed
	CUDA_CHECK(cudaStreamSynchronize(stream));

	// Free device memory

	CUDA_CHECK(cudaFree(d_t850));
	CUDA_CHECK(cudaFree(d_t700));
	CUDA_CHECK(cudaFree(d_t500));
	CUDA_CHECK(cudaFree(d_td850));
	CUDA_CHECK(cudaFree(d_td700));
	CUDA_CHECK(cudaFree(d_ki));
	CUDA_CHECK(cudaFree(d_vti));
	CUDA_CHECK(cudaFree(d_cti));
	CUDA_CHECK(cudaFree(d_tti));

	CUDA_CHECK(cudaStreamDestroy(stream));  // this blocks
}

void himan::plugin::stability_cuda::Prepare(const double* source, double** devptr, size_t memsize, cudaStream_t& stream)
{
	assert(source);
	CUDA_CHECK(cudaMalloc((void**)devptr, memsize));
	CUDA_CHECK(cudaMemcpyAsync((void*)*devptr, (const void*)source, memsize, cudaMemcpyHostToDevice, stream));
}
