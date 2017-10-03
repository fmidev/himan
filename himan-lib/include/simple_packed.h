/**
 * @file simple_packed.h
 *
 *
 * @brief Container to hold GRIB simple packed data.
 *
 * This class inherits from packed_data.
 */

#ifndef SIMPLE_PACKED_H
#define SIMPLE_PACKED_H

#ifdef HAVE_CUDA

#include "packed_data.h"

namespace himan
{
typedef packing_coefficients simple_packed_coefficients;

struct simple_packed : packed_data
{
	CUDA_HOST
	simple_packed() : packed_data() { packingType = kSimplePacking; }
	CUDA_HOST
	simple_packed(int theBitsPerValue, double theBinaryScaleFactor, double theDecimaleScaleFactor,
	              double theReferenceValue);

	CUDA_HOST
	simple_packed(const simple_packed& other);

	CUDA_HOST virtual ~simple_packed() {}
	virtual std::string ClassName() const { return "simple_packed"; }
	/**
	 * @brief Function will unpack binary array (unsigned char) to double array.
	 *
	 * Note! Argument d_arr should point to DEVICE MEMORY.
	 *
	 * @param d_arr Pointer to pre-allocated device memory
	 * @param N Length of allocated memory (not size)
	 * @param stream Cuda stream for execution. If 0 is given, function will create a temporary stream.
	 */

	virtual void Unpack(double* d_arr, size_t N, cudaStream_t* stream);

	CUDA_HOST
	void Pack(double* d_arr, size_t N, cudaStream_t* stream);

#ifdef __CUDACC__
	// Functions that are only visible for nvcc compiler

	CUDA_DEVICE
	void UnpackUnevenBytes(double* __restrict__ d_u, int idx);

	CUDA_DEVICE
	void UnpackFullBytes(double* __restrict__ d_u, int idx);

	template <typename T>
	CUDA_HOST T Min(T* d_arr, size_t N, cudaStream_t& stream);

	template <typename T>
	CUDA_HOST T Max(T* d_arr, size_t N, cudaStream_t& stream);
#endif
};

namespace simple_packed_util
{
__global__ void Unpack(unsigned char* d_p, double* d_u, int* d_b, himan::simple_packed_coefficients coeff,
                       bool hasBitmap, size_t N);

__device__ void UnpackUnevenBytes(unsigned char* __restrict__ d_p, double* __restrict__ d_u, int* __restrict__ d_b,
                                  himan::simple_packed_coefficients coeff, bool hasBitmap, int idx);

__device__ void UnpackFullBytes(unsigned char* __restrict__ d_p, double* __restrict__ d_u, int* __restrict__ d_b,
                                himan::simple_packed_coefficients coeff, bool hasBitmap, int idx);

__device__ void GetBitValue(unsigned char* p, long bitp, int* val);

__global__ void Pack(unsigned char* d_p, double* d_u, int* d_b, simple_packed_coefficients coeff, bool hasBitmap,
                     size_t N);

__device__ void PackFullBytes(unsigned char* __restrict__ d_p, const double* __restrict__ d_u, size_t values_len,
                              simple_packed_coefficients coeff, int idx);

__device__ void PackUnevenBytes(unsigned char* __restrict__ d_p, const double* __restrict__ d_u, size_t values_len,
                                simple_packed_coefficients coeff, int idx);

__device__ void SetBitOn(unsigned char* p, long bitp);

__device__ void SetBitOff(unsigned char* p, long bitp);
};

inline CUDA_HOST simple_packed::simple_packed(int theBitsPerValue, double theBinaryScaleFactor,
                                              double theDecimalScaleFactor, double theReferenceValue)
{
	coefficients.bitsPerValue = theBitsPerValue;
	coefficients.binaryScaleFactor = theBinaryScaleFactor;
	coefficients.decimalScaleFactor = theDecimalScaleFactor;
	coefficients.referenceValue = theReferenceValue;
	packingType = kSimplePacking;
}

inline CUDA_HOST simple_packed::simple_packed(const simple_packed& other) : packed_data(other)
{
	coefficients.bitsPerValue = other.coefficients.bitsPerValue;
	coefficients.binaryScaleFactor = other.coefficients.binaryScaleFactor;
	coefficients.decimalScaleFactor = other.coefficients.decimalScaleFactor;
	coefficients.referenceValue = other.coefficients.referenceValue;
	packingType = kSimplePacking;
}

}  // namespace himan

#endif /* HAVE_CUDA */
#endif /* SIMPLE_PACKED_H */
