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
