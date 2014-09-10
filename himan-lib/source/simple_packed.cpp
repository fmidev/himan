/**
 * @file simple_packed.cpp
 *
 * @date Feb 18, 2014
 * @author partio
 */

#include "simple_packed.h"
#include "cuda_helper.h"

#ifdef HAVE_CUDA

void himan::simple_packed::Unpack(double* arr, size_t N)
{
	if (!packedLength)
	{
		return;
	}

	assert(N == unpackedLength);

	double* d_arr = 0;

	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **> (&d_arr), sizeof(double) * N));

	Unpack(d_arr, N, 0); // use 0-stream

	CUDA_CHECK(cudaMemcpy(arr, d_arr, sizeof(double) * N, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(d_arr));

}

#endif