/**
 * @file simple_packed.cu
 *
 * @date Aug 27, 2013
 * @author partio
 */

#include "simple_packed.h"

#include "cuda_helper.h"

using namespace himan;

__host__
void simple_packed::Unpack(double* d_u, cudaStream_t* stream)
{
	int blockSize = 512;
	int gridSize = unpackedLength / blockSize + (unpackedLength % blockSize == 0 ? 0 : 1);
	
	unsigned char* d_p = 0; // device-packed data
	int* d_b = 0; // device-bitmap

	// These are allocated with cudaHostAlloc zero-copy pinned memory
	CUDA_CHECK(cudaHostGetDevicePointer(&d_p, data, 0));

	if (HasBitmap())
	{
		CUDA_CHECK(cudaHostGetDevicePointer(&d_b, bitmap, 0));
	}
	
	simple_packed_util::Unpack <<< gridSize, blockSize, 0, *stream >>> (d_p, d_u, d_b, coefficients, HasBitmap(), unpackedLength);

}

__global__
void simple_packed_util::Unpack(unsigned char* d_p, double* d_u, int* d_b, simple_packed_coefficients coeff, bool hasBitmap, size_t N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		if (coeff.bitsPerValue % 8)
		{
			UnpackUnevenBytes(d_p, d_u, d_b, coeff, hasBitmap, idx);
		}
		else
		{
			UnpackFullBytes(d_p, d_u, d_b, coeff, hasBitmap, idx);
		}
	}
}

__device__
void simple_packed_util::GetBitValue(unsigned char* p, long bitp, int *val)
{
	p += (bitp >> 3);
	*val = (*p&(1<<(7-(bitp%8))));
}

__device__
void simple_packed_util::UnpackFullBytes(unsigned char* __restrict__ d_p, double* __restrict__ d_u, int* __restrict__ d_b, simple_packed_coefficients coeff, bool hasBitmap, int idx)
{
	int bc;
	unsigned long lvalue;

	int l = coeff.bitsPerValue / 8;

	int bm = idx;
	int value_found = 1;

	if (hasBitmap)
	{
		bm = d_b[idx];

		if (bm == 0)
		{
			d_u[idx] = kFloatMissing;
			value_found = 0;
		}
		else
		{
			bm--;
		}
	}

	if (value_found)
	{
		size_t o = bm*l;

		lvalue	= 0;
		lvalue	<<= 8;
		lvalue |= d_p[o++] ;

		for ( bc=1; bc<l; bc++ )
		{
			lvalue <<= 8;
			lvalue |= d_p[o++] ;
		}

		d_u[idx] = ((lvalue * coeff.binaryScaleFactor) + coeff.referenceValue) * coeff.decimalScaleFactor;
	}
}

__device__
void simple_packed_util::UnpackUnevenBytes(unsigned char* __restrict__ d_p, double* __restrict__ d_u, int* __restrict__ d_b, simple_packed_coefficients coeff, bool hasBitmap, int idx)
{
	int j=0;
	unsigned long lvalue;

	int bm = idx;
	int value_found = 1;

	/*
	 * Check if bitmap is set.
	 * If bitmap is set and indicates that value for this element is missing, do
	 * not proceed to calculating phase.
	 *
	 * If bitmap is set and indicates that value exists for this element, the index
	 * for the actual data is the one indicated by the bitmap array. From this index
	 * we reduce one (1) because that one is added to the value in unpack_bitmap.
	 */

	if (hasBitmap)
	{
		bm = d_b[idx];

		if (bm == 0)
		{
			d_u[idx] = kFloatMissing;
			value_found = 0;
		}
		else
		{
			bm--;
		}
	}

	if (value_found)
	{
		long bitp=coeff.bitsPerValue*bm;

		lvalue=0;

		for(j=0; j< coeff.bitsPerValue; j++)
		{
			lvalue <<= 1;
			int val;

			GetBitValue(d_p, bitp, &val);

			if (val) lvalue += 1;

			bitp += 1;
		}

		d_u[idx] = ((lvalue * coeff.binaryScaleFactor) + coeff.referenceValue) * coeff.decimalScaleFactor;
	}

}