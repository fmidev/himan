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
struct packed_data 
{
       	bool HasData() const { return false; }
};
}

#else

#ifdef __CUDACC__
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#else
#define CUDA_HOST
#define CUDA_DEVICE
#endif

#include <cuda_runtime_api.h>
#include "himan_common.h"

namespace himan
{

struct packed_data
{

	CUDA_HOST
	packed_data() : data(0), packedLength(0), unpackedLength(0), bitmap(0), bitmapLength(0), packingType(kUnknownPackingType) {}

	CUDA_HOST CUDA_DEVICE
	virtual ~packed_data();

	/**
	 * @brief Copy constructor for packed data
	 *
	 * This is defined for both gcc and nvcc separately
	 * 
     * @param other packed_data instance that we are copying from
     */
	
	CUDA_HOST
	packed_data(const packed_data& other);

	virtual std::string ClassName() const { return "packed_data"; }

	void Resize(size_t newPackedLength, size_t newUnpackedLength);
	void Set(unsigned char* packedData, size_t packedDataLength, size_t unpackedDataLength);
	void Bitmap(int* newBitmap, size_t newBitmapLength);
	void Clear();
	
	CUDA_HOST
	bool HasData() const;

	CUDA_HOST CUDA_DEVICE
	bool HasBitmap() const;

	unsigned char* data;
	size_t packedLength;
	size_t unpackedLength;
	int* bitmap;
	size_t bitmapLength;

	HPPackingType packingType;

};

inline
CUDA_HOST
packed_data::packed_data(const packed_data& other)
	: packedLength(other.packedLength)
	, unpackedLength(other.unpackedLength)
	, packingType(other.packingType)
{

	data = 0;
	bitmap = 0;

	if (other.packedLength)
	{

#ifndef __CUDACC__
		cudaHostAlloc(reinterpret_cast<void**> (&data), packedLength * sizeof(unsigned char), cudaHostAllocMapped);

		memcpy(data, other.data, packedLength * sizeof(unsigned char));
#endif

	}

	if (other.bitmapLength)
	{

#ifndef __CUDACC__
		cudaHostAlloc(reinterpret_cast<void**> (&bitmap), bitmapLength * sizeof(int), cudaHostAllocMapped);

		memcpy(bitmap, other.bitmap, bitmapLength * sizeof(int));
#endif
	}
}

inline
CUDA_HOST CUDA_DEVICE
packed_data::~packed_data()
{
#ifndef __CUDACC__
	Clear();
#endif
}

inline
CUDA_HOST
bool packed_data::HasData() const
{
	return (packedLength > 0);
}

inline
CUDA_HOST CUDA_DEVICE
bool packed_data::HasBitmap() const
{
	return (bitmapLength > 0);
}

inline
void packed_data::Clear()
{
	if (packedLength)
	{
		cudaFreeHost(data);
		packedLength = 0;
		unpackedLength = 0;
	}

	if (bitmapLength)
	{
		cudaFreeHost(bitmap);
		bitmapLength = 0;
	}
}

} // namespace himan

#endif  /* HAVE_CUDA */
#endif	/* PACKED_DATA_H */
