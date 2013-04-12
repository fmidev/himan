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
 * Default behavior with regards to copying structs is as follows:
 *
 * WHEN STRUCT IS COPIED AT HOST SIDE, WITH HOST COMPILER (g++):
 * - struct is deep-copied
 *
 * WHEN STRUCT IS COPIED AT DEVICE SIDE, WITH CUDA COMPILER (nvcc):
 * - struct is shallow-copied
 *
 * From this follows that creating and freeing (destrying) struct MUST BE DONE ON THE HOST SIDE.
 * 
 * Explanation for this is that since the struct is passed by-value from host to
 * device (device cannot access data in host memory, therefore it must be copied), we don't
 * want to copy the array that holds the data since that's already memcopied using
 * cudaMemcpy -- we only want to copy the metadata.
 * 
 * So when struct is passed to and fro in the device code, only shallow copies are
 * done.
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
	packed_data() : data(0), dataLength(0), packingType(kUnknownPackingType) {}

#ifdef __CUDACC__
__device__  __host__
#endif
	virtual ~packed_data();
	
	packed_data(const packed_data& other);

	virtual std::string ClassName() const { return "packed_data"; }

	void Resize(size_t newDataLength);
	void Set(unsigned char* newData, size_t newDataLength);
	void Clear();

	unsigned char* data;
	size_t dataLength;
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

	if (other.data)
	{
#ifndef __CUDA_ARCH__
		dataLength = other.dataLength;

		cudaMallocHost(reinterpret_cast<void**> (&data), dataLength * sizeof(unsigned char));

		memcpy(data, other.data, dataLength * sizeof(unsigned char));
#endif
	}
	else
	{
		data = 0;
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
 __device__ __host__
#endif
void packed_data::Clear()
{
	if (data)
	{
#ifndef __CUDA_ARCH__
		cudaFreeHost(data);
#endif
	}
}

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

	int bitsPerValue;
	double binaryScaleFactor;
	double decimalScaleFactor;
	double referenceValue;
};


inline 
#ifdef __CUDACC__
__device__  __host__
#endif
simple_packed::simple_packed(int theBitsPerValue, double theBinaryScaleFactor, double theDecimalScaleFactor, double theReferenceValue)
	: bitsPerValue(theBitsPerValue)
	, binaryScaleFactor(theBinaryScaleFactor)
	, decimalScaleFactor(theDecimalScaleFactor)
	, referenceValue(theReferenceValue)
	
{
	packingType = kSimplePacking;
}

inline
#ifdef __CUDACC__
__device__  __host__
#endif
simple_packed::simple_packed(const simple_packed& other)
	: packed_data(other)
	, bitsPerValue(other.bitsPerValue)
	, binaryScaleFactor(other.binaryScaleFactor)
	, decimalScaleFactor(other.decimalScaleFactor)
	, referenceValue(other.referenceValue)
	
{}

} // namespace himan

#endif  /* HAVE_CUDA */
#endif	/* PACKED_DATA_H */
