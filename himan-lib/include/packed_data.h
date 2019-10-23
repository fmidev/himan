/**
 * @file packed_data.h
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

#pragma once

#ifndef HAVE_CUDA
// Define shells so that compilation succeeds
#include "serialization.h"
namespace himan
{
struct packed_data
{
	bool HasData() const
	{
		return false;
	}

   private:
#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
	}
#endif
};

#else

#include "cuda_helper.h"
#include "himan_common.h"
#include <NFmiGribPacking.h>
#include <stdexcept>

namespace himan
{
struct packed_data
{
	packed_data() = default;

	CUDA_HOST virtual ~packed_data()
	{
		Clear();
	}

	/**
	 * @brief Copy constructor for packed data
	 *
	 * This is defined for both gcc and nvcc separately
	 *
	 * @param other packed_data instance that we are copying from
	 */

	CUDA_HOST
	packed_data(const packed_data& other);

	void Clear();

	CUDA_HOST
	bool HasData() const
	{
		return (unpackedLength > 0);
	}

	CUDA_HOST CUDA_DEVICE bool HasBitmap() const
	{
		return (bitmapLength > 0);
	}

	unsigned char* data = nullptr;
	size_t packedLength = 0;
	size_t unpackedLength = 0;
	int* bitmap = nullptr;
	size_t bitmapLength = 0;

	HPPackingType packingType = kUnknownPackingType;
};

inline packed_data::packed_data(const packed_data& other)
    : packedLength(other.packedLength),
      unpackedLength(other.unpackedLength),
      bitmapLength(other.bitmapLength),
      packingType(other.packingType)
{
	data = nullptr;
	bitmap = nullptr;

	if (other.packedLength)
	{
		CUDA_CHECK(
		    cudaHostAlloc(reinterpret_cast<void**>(&data), packedLength * sizeof(unsigned char), cudaHostAllocMapped));

		memcpy(data, other.data, packedLength * sizeof(unsigned char));
	}

	if (other.bitmapLength)
	{
		cudaHostAlloc(reinterpret_cast<void**>(&bitmap), bitmapLength * sizeof(int), cudaHostAllocMapped);

		memcpy(bitmap, other.bitmap, bitmapLength * sizeof(int));
	}
}

inline void packed_data::Clear()
{
	if (data)
	{
		CUDA_CHECK(cudaFreeHost(data));
	}

	packedLength = 0;
	data = nullptr;

	if (bitmap)
	{
		CUDA_CHECK(cudaFreeHost(bitmap));
	}

	bitmapLength = 0;
	bitmap = nullptr;

	unpackedLength = 0;
}

struct simple_packed : public packed_data
{
	CUDA_HOST
	simple_packed() : packed_data()
	{
		packingType = kSimplePacking;
	}

	CUDA_HOST
	simple_packed(int theBitsPerValue, double theBinaryScaleFactor, double theDecimaleScaleFactor,
	              double theReferenceValue);

	simple_packed(const simple_packed& other) = default;

	NFmiGribPacking::packing_coefficients coefficients;
};

inline CUDA_HOST simple_packed::simple_packed(int theBitsPerValue, double theBinaryScaleFactor,
                                              double theDecimalScaleFactor, double theReferenceValue)
    : simple_packed()
{
	coefficients.bitsPerValue = theBitsPerValue;
	coefficients.binaryScaleFactor = theBinaryScaleFactor;
	coefficients.decimalScaleFactor = theDecimalScaleFactor;
	coefficients.referenceValue = theReferenceValue;
}

#endif

}  // namespace himan
