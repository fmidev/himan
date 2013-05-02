/**
 * @file packed_data.h
 * @author partio
 *
 * @date April 5, 2013, 10:36 PM
 *
 * @brief Container to hold packed data.
 *
 * Data is later on unpacked in GPU, so we have CUDA specific functions (this should be always
 * the case since it makes no sense to unpack it in host CPU outside grib_api)
 *
 * All CUDA commands are still wrapped with preprocessor macros so that HIMAN can be compiled
 * even on a machine that does not have CUDA SDK installed.
 *
 */

#ifndef PACKED_DATA_H
#define	PACKED_DATA_H

#ifndef HAVE_CUDA

// Define shells so that compilation succeeds
namespace himan
{
	struct packed_data {};
}

#else

#include <cuda_runtime_api.h>
#include "himan_common.h"

namespace himan
{

struct packed_data
{

#ifdef __CUDACC__
__device__  __host__
#endif
	packed_data() : data(0), dataLength(0), bitmap(0), bitmapLength(0), packingType(kUnknownPackingType) {}

#ifdef __CUDACC__
__device__  __host__
#endif
	virtual ~packed_data();
	
	packed_data(const packed_data& other);

	virtual std::string ClassName() const { return "packed_data"; }

	void Resize(size_t newDataLength);
	void Set(unsigned char* newData, size_t newDataLength);
	void Bitmap(int* newBitmap, size_t newBitmapLength);
	void Clear();
	
#ifdef __CUDACC__
__device__  __host__
#endif
	bool HasData() const;

#ifdef __CUDACC__
__device__  __host__
#endif
	bool HasBitmap() const;

	unsigned char* data;
	size_t dataLength;
	int* bitmap;
	size_t bitmapLength;

	HPPackingType packingType;

};

inline
#ifdef __CUDACC__
__device__  __host__
#endif
packed_data::packed_data(const packed_data& other)
	: dataLength(other.dataLength)
	, packingType(other.packingType)
{

	if (other.dataLength)
	{
#ifndef __CUDACC__
		dataLength = other.dataLength;

		//cudaMallocHost(reinterpret_cast<void**> (&data), dataLength * sizeof(unsigned char));
		cudaHostAlloc(reinterpret_cast<void**> (&data), dataLength * sizeof(unsigned char), cudaHostAllocMapped);

		memcpy(data, other.data, dataLength * sizeof(unsigned char));
#endif
	}
	else
	{
		data = 0;
	}

	if (other.bitmapLength)
	{
#ifndef __CUDACC__
		bitmapLength = other.bitmapLength;
		
		cudaHostAlloc(reinterpret_cast<void**> (&bitmap), bitmapLength * sizeof(int), cudaHostAllocMapped);

		memcpy(bitmap, other.bitmap, bitmapLength * sizeof(int));
#endif
	}
	else
	{
		bitmap = 0;
	}
}

inline
#ifdef __CUDACC__
__device__  __host__
#endif
packed_data::~packed_data()
{
#ifndef __CUDACC__
	Clear();
#endif
}

inline
#ifdef __CUDACC__
__device__  __host__
#endif
bool packed_data::HasData() const
{
	return (dataLength > 0);
}

inline
#ifdef __CUDACC__
__device__  __host__
#endif
bool packed_data::HasBitmap() const
{
	return (bitmapLength > 0);
}

inline
#ifdef __CUDACC__
 __device__ __host__
#endif
void packed_data::Clear()
{
	if (dataLength)
	{
#ifndef __CUDACC__
		cudaFreeHost(data);
		dataLength = 0;
#endif
	}

	if (bitmapLength)
	{
#ifndef __CUDACC__
		cudaFreeHost(bitmap);
		bitmapLength = 0;
#endif
	}
}

struct simple_packed_coefficients
{
	int bitsPerValue;
	double binaryScaleFactor;
	double decimalScaleFactor;
	double referenceValue;

#ifdef __CUDACC__
	__host__ __device__
#endif
	simple_packed_coefficients()
		: bitsPerValue(0), binaryScaleFactor(0), decimalScaleFactor(0), referenceValue(0)
	{}

};

struct simple_packed : packed_data
{
#ifdef __CUDACC__
	__device__ __host__
#endif
	simple_packed() : packed_data()
	{
		packingType = kSimplePacking;
	}

#ifdef __CUDACC__
	__device__ __host__
#endif
	simple_packed(int theBitsPerValue, double theBinaryScaleFactor, double theDecimaleScaleFactor, double theReferenceValue);

#ifdef __CUDACC__
	__device__ __host__
#endif
	simple_packed(const simple_packed& other);

#ifdef __CUDACC__
	__device__ __host__
#endif
	virtual ~simple_packed() {}

	virtual std::string ClassName() const { return "simple_packed"; }
	simple_packed_coefficients coefficients;

};


inline 
#ifdef __CUDACC__
__device__  __host__
#endif
simple_packed::simple_packed(int theBitsPerValue, double theBinaryScaleFactor, double theDecimalScaleFactor, double theReferenceValue) 
{
	coefficients.bitsPerValue = theBitsPerValue;
	coefficients.binaryScaleFactor = theBinaryScaleFactor;
	coefficients.decimalScaleFactor = theDecimalScaleFactor;
	coefficients.referenceValue = theReferenceValue;
	packingType = kSimplePacking;
}

inline
#ifdef __CUDACC__
__device__  __host__
#endif
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
#endif	/* PACKED_DATA_H */
