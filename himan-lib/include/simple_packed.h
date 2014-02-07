/**
 * @file simple_packed.h
 * @author partio
 *
 * @date August 30, 2013, 08:28 AM
 *
 * @brief Container to hold GRIB simple packed data.
 *
 * This class inherits from packed_data.
 */

#ifndef SIMPLE_PACKED_H
#define	SIMPLE_PACKED_H

#ifdef HAVE_CUDA

#include "packed_data.h"

namespace himan
{

struct simple_packed_coefficients
{
	int bitsPerValue;
	double binaryScaleFactor;
	double decimalScaleFactor;
	double referenceValue;

	CUDA_HOST
	simple_packed_coefficients()
		: bitsPerValue(0), binaryScaleFactor(0), decimalScaleFactor(0), referenceValue(0)
	{}

};

struct simple_packed : packed_data
{
	CUDA_HOST
	simple_packed() : packed_data()
	{
		packingType = kSimplePacking;
	}

	CUDA_HOST
	simple_packed(int theBitsPerValue, double theBinaryScaleFactor, double theDecimaleScaleFactor, double theReferenceValue);

	CUDA_HOST
	simple_packed(const simple_packed& other);

	CUDA_HOST CUDA_DEVICE
	virtual ~simple_packed() {}

	virtual std::string ClassName() const { return "simple_packed"; }

#ifdef __CUDACC__
	// Functions that are only visible for nvcc compiler

	/**
	 * @brief Function will unpack binary array (unsigned char) to double array.
	 *
	 * Function is synchronous due to implicit synchronization caused by cudaFree().
	 *
	 * Note! Return pointer to double array RESIDES IN DEVICE MEMORY. The caller
	 * is responsible for managing the memory.
	 *
	 */

	__host__
	double* Unpack(cudaStream_t* stream);

	__device__
	void UnpackUnevenBytes(double* __restrict__ d_u, int idx);

	__device__
	void UnpackFullBytes(double* __restrict__ d_u, int idx);
#endif
	
	simple_packed_coefficients coefficients;

};

namespace simple_packed_util
{
__global__
void Unpack(unsigned char* d_p, double* d_u, int* d_b, himan::simple_packed_coefficients coeff, bool hasBitmap, size_t N);

__device__
void UnpackUnevenBytes(unsigned char* __restrict__ d_p, double* __restrict__ d_u, int* __restrict__ d_b, himan::simple_packed_coefficients coeff, bool hasBitmap, int idx);

__device__
void UnpackFullBytes(unsigned char* __restrict__ d_p, double* __restrict__ d_u, int* __restrict__ d_b, himan::simple_packed_coefficients coeff, bool hasBitmap, int idx);

__device__
void GetBitValue(unsigned char* p, long bitp, int *val);

};

inline 
CUDA_HOST
simple_packed::simple_packed(int theBitsPerValue, double theBinaryScaleFactor, double theDecimalScaleFactor, double theReferenceValue) 
{
	coefficients.bitsPerValue = theBitsPerValue;
	coefficients.binaryScaleFactor = theBinaryScaleFactor;
	coefficients.decimalScaleFactor = theDecimalScaleFactor;
	coefficients.referenceValue = theReferenceValue;
	packingType = kSimplePacking;
}

inline
CUDA_HOST
simple_packed::simple_packed(const simple_packed& other)
	: packed_data(other)
{
	coefficients.bitsPerValue = other.coefficients.bitsPerValue;
	coefficients.binaryScaleFactor = other.coefficients.binaryScaleFactor;
	coefficients.decimalScaleFactor = other.coefficients.decimalScaleFactor;
	coefficients.referenceValue = other.coefficients.referenceValue;
	packingType = kSimplePacking;

}

} // namespace himan

#endif  /* HAVE_CUDA */
#endif	/* SIMPLE_PACKED_H */
