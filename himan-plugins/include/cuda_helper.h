#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda_runtime.h>
//#include <cuda.h>

const float kFloatMissing = 32700.f;

void CheckCudaError(cudaError_t errarg, const char* file, const int line);
void CheckCudaErrorString(const char* errstr, const char* file,	const int line);

__device__ void SimpleUnpackFullBytes(const unsigned char* __restrict__ d_p,double* __restrict__ d_u, size_t values_len, int bpv, double bsf, double dsf, double rv);
__device__ void SimpleUnpackUnevenBytes(unsigned char* __restrict__ d_p, double* __restrict__ d_u, size_t values_len, int bpv, double bsf, double dsf, double rv);
__device__ void GetBitValue(const unsigned char* d_p, long bitp, int *val);

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

__device__ void SimpleUnpackFullBytes(const unsigned char* __restrict__ d_p,
											double* __restrict__ d_u,
											size_t values_len, int bpv, double bsf, double dsf, double rv)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int bc;
	unsigned long lvalue;

	int l = bpv/8;
	size_t o = idx*l;

	if (idx < values_len)
	{
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

__device__ void SimpleUnpackUnevenBytes(unsigned char* __restrict__ d_p,
											double* __restrict__ d_u,
											size_t values_len, int bpv, double bsf, double dsf, double rv)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int j=0;
	unsigned long lvalue;
	long bitp=bpv*idx;

	if (idx < values_len)
	{
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

__device__ void GetBitValue(const unsigned char* d_p, long bitp, int *val)
{
	d_p += (bitp >> 3);
	*val = (*d_p&(1<<(7-(bitp%8))));
}



#endif
