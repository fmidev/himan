#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#ifdef HAVE_CUDA

void CheckCudaError(cudaError_t errarg, const char* file, const int line);
void CheckCudaErrorString(const char* errstr, const char* file,	const int line);

#define CUDA_CHECK(errarg)	 CheckCudaError(errarg, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR_MSG(errstr) CheckCudaErrorString(errstr, __FILE__, __LINE__)

inline void CheckCudaError(cudaError_t errarg, const char* file, const int line)
{
	if(errarg)
	{
		std::cerr << "Error at " << file << "(" << line << "): " << cudaGetErrorString(errarg) << std::endl;
		exit(1);
	}
}


inline void CheckCudaErrorString(const char* errstr, const char* file,	const int line)
{
	cudaError_t err = cudaGetLastError();

	if(err != cudaSuccess)
	{
		std::cerr	<< "Error: "
					<< errstr
					<< " "
					<< file 
					<< " at ("
					<< line
					<< "): "
					<< cudaGetErrorString(err)
					<< std::endl;
		
		exit(1);
	}
}

#ifdef __CUDACC__

#include "packed_data.h"

namespace himan
{
	
const float kFloatMissing = 32700.f;

#define BitMask1(i)	(1u << i)
#define BitTest(n,i)	!!((n) & BitMask1(i))

inline __host__ void UnpackBitmap(const unsigned char* __restrict__ bitmap, int* unpacked, size_t len)
{
	size_t i, v = 1, idx = 0;
	short j = 0;

	for (i = 0; i < len; i++)
	{
		for (j = 7; j >= 0; j--)
		{
			if (BitTest(bitmap[i], j))
			{
				unpacked[idx] = v++;
			}
      idx++;
    }
  }
}

inline __device__ void GetBitValue(const unsigned char* p, long bitp, int *val)
{
	p += (bitp >> 3);
	*val = (*p&(1<<(7-(bitp%8))));
}

inline __device__ void SetBitOn(unsigned char* p, long bitp)
{
  p += bitp/8;
  *p |= (1u << (7-((bitp)%8)));
}

inline __device__ void SetBitOff( unsigned char* p, long bitp)
{
  p += bitp/8;
  *p &= ~(1u << (7-((bitp)%8)));
}

inline __device__ void SimpleUnpackFullBytes(const unsigned char* __restrict__ d_p,
											double* __restrict__ d_u,
											const int* __restrict__ d_bm,
											size_t values_len,
											int bpv, double bsf, double dsf, double rv, bool hasBitmap, int idx)
{
	int bc;
	unsigned long lvalue;

	int l = bpv/8;

	if (idx < values_len)
	{
		int bm = idx;
		int value_found = 1;

		if (hasBitmap)
		{
			bm = d_bm[idx];

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

			d_u[idx] = ((lvalue*bsf)+rv)*dsf;
		}
	}
}

inline __device__ void SimpleUnpackUnevenBytes(const unsigned char* __restrict__ d_p,
												double* __restrict__ d_u,
												const int* __restrict__ d_bm,
												size_t values_len, int bpv, double bsf, double dsf, double rv, bool hasBitmap, int idx)
{
	int j=0;
	unsigned long lvalue;

	if (idx < values_len)
	{
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
			bm = d_bm[idx];

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
			long bitp=bpv*bm;

			lvalue=0;

			for(j=0; j< bpv;j++)
			{
				lvalue <<= 1;
				int val;

				GetBitValue(d_p, bitp, &val);

				if (val) lvalue += 1;

				bitp += 1;
			}
			d_u[idx] = ((lvalue*bsf)+rv)*dsf;
		}
	}
}

__global__ void SimpleUnpack(const unsigned char* d_p, double* d_u, const int* d_bm, himan::simple_packed_coefficients coeff, size_t N, bool hasBitmap)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		if (coeff.bitsPerValue % 8)
		{
			SimpleUnpackUnevenBytes(d_p, d_u, d_bm, N,coeff.bitsPerValue, coeff.binaryScaleFactor, coeff.decimalScaleFactor, coeff.referenceValue, hasBitmap, idx);
		}
		else
		{
			SimpleUnpackFullBytes(d_p, d_u, d_bm, N, coeff.bitsPerValue, coeff.binaryScaleFactor, coeff.decimalScaleFactor, coeff.referenceValue, hasBitmap, idx);
		}
	}
}

// ----------- NOTE -----------
// PACKING FUNCTION DO NOT WORK YET

inline __device__ void SimplePackUnevenBytes(const double* d_u, unsigned char* d_p,
									size_t values_len, int bpv, double bsf, double dsf, double rv, int idx)
{

	double x=(((d_u[idx]*dsf)-rv)*bsf)+0.5;
	long bitp=bpv*idx;

	long  i = 0;

	for (i=bpv-1; i >= 0; i--)
	{
		if(BitTest(static_cast<unsigned long> (x),i))
		{
			SetBitOn(d_p, bitp);
		}
		else
		{
			SetBitOff(d_p, bitp);
		}
	}
}

inline __device__ void SimplePackFullBytes(const double* d_u, unsigned char* d_p,
									size_t values_len, int bpv, double bsf, double dsf, double rv, int idx)
{

	int blen=0;
	unsigned char* encoded = d_p + idx * static_cast<int> (bpv/8);

	blen = bpv;
	double x = ((((d_u[idx]*dsf)-rv)*bsf)+0.5);
	unsigned long unsigned_val = (unsigned long)x;
	while(blen >= 8)
	{
		blen -= 8;
		*encoded = (unsigned_val >> blen);
		encoded++;
		//*off += idx*8;

	}
}

inline __device__ void SimplePack(const double* d_u, unsigned char* d_p,
									size_t values_len, int bpv, double bsf, double dsf, double rv, int idx)
{

	if (bpv%8)
	{
		SimplePackUnevenBytes(d_u, d_p, values_len, bpv, bsf, dsf, rv, idx);
	}
	else
	{
		SimplePackFullBytes(d_u, d_p, values_len, bpv, bsf, dsf, rv, idx);
	}
}

} // namespace himan

#endif // __CUDACC__
#endif // HAVE_CUDA
#endif // CUDA_HELPER_H
