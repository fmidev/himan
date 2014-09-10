/**
 * @file simple_packed.cu
 *
 * @date Aug 27, 2013
 * @author partio
 */

#include "simple_packed.h"

#include "cuda_helper.h"

#define BitTest(n,i)	!!((n) & BitMask1(i))
#define BitMask1(i)	(1u << i)

using namespace himan;

__host__
double* simple_packed::Unpack(double* d_arr, cudaStream_t* stream)
{
	if (!packedLength)
	{
		return 0;
	}

	// We need to create a stream if no stream is specified since dereferencing
	// a null pointer is, well, not a good thing.

	bool destroyStream = false;
	
	if (!stream)
	{
		stream = new cudaStream_t;
		CUDA_CHECK(cudaStreamCreate(stream));
		destroyStream = true;
	}

	int blockSize = 512;
	int gridSize = unpackedLength / blockSize + (unpackedLength % blockSize == 0 ? 0 : 1);

	unsigned char*	d_p = 0; // device-packed data
	int*			d_b = 0; // device-bitmap

	CUDA_CHECK(cudaMalloc((void**) (&d_p), packedLength * sizeof(unsigned char)));
	CUDA_CHECK(cudaMemcpyAsync(d_p, data, packedLength * sizeof(unsigned char), cudaMemcpyHostToDevice, *stream));

	if (HasBitmap())
	{
		CUDA_CHECK(cudaMalloc((void**) (&d_b), bitmapLength * sizeof(int)));
		CUDA_CHECK(cudaMemcpyAsync(d_b, bitmap, bitmapLength * sizeof(int), cudaMemcpyHostToDevice, *stream));
		CUDA_CHECK(cudaStreamSynchronize(*stream));
	}

	simple_packed_util::Unpack <<< gridSize, blockSize, 0, *stream >>> (d_p, d_arr, d_b, coefficients, HasBitmap(), unpackedLength);

	CUDA_CHECK(cudaFree(d_p));

	if (HasBitmap())
	{
		CUDA_CHECK(cudaFree(d_b));
	}

	if (destroyStream)
	{
		CUDA_CHECK(cudaStreamDestroy(*stream));
		delete stream;
	}
}

__global__
void simple_packed_util::Unpack(unsigned char* d_p, double* d_u, int* d_b, simple_packed_coefficients coeff, bool hasBitmap, size_t N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		if (coeff.bitsPerValue % 8) // modulo is expensive but "Compiler will convert literal power-of-2 divides to bitwise shifts"
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
double GetGribPower(long s,long n)
{
	double divisor = 1.0;
	while(s < 0)
	{
		divisor /= n;
		s++;
	}
	while(s > 0)
	{
		divisor *= n;
		s--;
	}
	return divisor;
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

__device__
void simple_packed_util::PackUnevenBytes(unsigned char* __restrict__ d_p, const double* __restrict__ d_u,
									size_t values_len, simple_packed_coefficients coeff, int idx)
{

	double decimal = GetGribPower(-coeff.decimalScaleFactor, 10);
	double divisor = GetGribPower(-coeff.binaryScaleFactor, 2);

	double x=(((d_u[idx]*decimal)-coeff.referenceValue)*divisor)+0.5;

	// long bitp = coeff.bitsPerValue * idx;

	long  i = 0;

	for (i=coeff.bitsPerValue-1; i >= 0; i--)
	{
		if(BitTest(static_cast<unsigned long> (x),i))
		{
			SetBitOn(d_p, i);
		}
		else
		{
			SetBitOff(d_p, i);
		}
	}
}

__device__
void simple_packed_util::PackFullBytes(unsigned char* __restrict__ d_p, const double* __restrict__ d_u,
									size_t values_len, simple_packed_coefficients coeff, int idx)
{

	double decimal = GetGribPower(-coeff.decimalScaleFactor, 10);
	double divisor = GetGribPower(-coeff.binaryScaleFactor, 2);
	
	// unsigned char* encoded = d_p + idx * static_cast<int> (coefficients.bpv/8);

	double x = ((((d_u[idx]*decimal)-coeff.referenceValue)*divisor)+0.5);
	unsigned long unsigned_val = (unsigned long)x;

	unsigned char* encoded = &d_p[idx];

	while(coeff.bitsPerValue >= 8)
	{
		coeff.bitsPerValue -= 8;
		*encoded = (unsigned_val >> coeff.bitsPerValue);
		encoded++;
	}
}


__global__
void simple_packed_util::Pack(unsigned char* d_p, double* d_u, int* d_b, simple_packed_coefficients coeff, bool hasBitmap, size_t N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		if (coeff.bitsPerValue % 8) // modulo is expensive but "Compiler will convert literal power-of-2 divides to bitwise shifts"
		{
			PackUnevenBytes(d_p, d_u, N, coeff, idx);
		}
		else
		{
			PackFullBytes(d_p, d_u, N, coeff, idx);
		}
	}
}

__device__
void simple_packed_util::SetBitOn(unsigned char* p, long bitp)
{
  p += bitp/8;
  *p |= (1u << (7-((bitp)%8)));
}

__device__
void simple_packed_util::SetBitOff( unsigned char* p, long bitp)
{
  p += bitp/8;
  *p &= ~(1u << (7-((bitp)%8)));
}

