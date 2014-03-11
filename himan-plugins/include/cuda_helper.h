#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <iostream>

#ifdef HAVE_CUDA

namespace himan
{
#ifdef __CUDACC__
const double kFloatMissing = 32700;
#endif

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
} // namespace himan
#if 0
#include "packed_data.h"

namespace himan
{

const float kFloatMissing = 32700.f;

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

} // namespace himan
#endif
#endif // HAVE_CUDA
#endif // CUDA_HELPER_H
