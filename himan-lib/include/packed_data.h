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

#ifndef PACKED_DATA_H
#define PACKED_DATA_H

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

#include "cuda_helper.h"
#include "himan_common.h"
#include <stdexcept>
#include <string>

namespace himan
{
struct packing_coefficients
{
	int bitsPerValue;
	double binaryScaleFactor;
	double decimalScaleFactor;
	double referenceValue;

	CUDA_HOST
	packing_coefficients() : bitsPerValue(0), binaryScaleFactor(0), decimalScaleFactor(0), referenceValue(0) {}
};

struct packed_data
{
	CUDA_HOST
	packed_data()
	    : data(0), packedLength(0), unpackedLength(0), bitmap(0), bitmapLength(0), packingType(kUnknownPackingType)
	{
	}

	CUDA_HOST CUDA_DEVICE virtual ~packed_data();

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

	CUDA_HOST CUDA_DEVICE bool HasBitmap() const;

	virtual void Unpack(double* d_arr, size_t N, cudaStream_t* stream)
	{
		throw std::runtime_error("top level Unpack called");
	}

	unsigned char* data;
	size_t packedLength;
	size_t unpackedLength;
	int* bitmap;
	size_t bitmapLength;

	HPPackingType packingType;

	packing_coefficients coefficients;
};

namespace packed_data_util
{
CUDA_HOST CUDA_DEVICE double GetGribPower(long s, long n);
}

inline CUDA_HOST CUDA_DEVICE double himan::packed_data_util::GetGribPower(long s, long n)
{
	double divisor = 1.0;
	double dn = static_cast<double>(n);

	while (s < 0)
	{
		divisor /= dn;
		s++;
	}
	while (s > 0)
	{
		divisor *= dn;
		s--;
	}
	return divisor;
}

inline CUDA_HOST CUDA_DEVICE packed_data::~packed_data()
{
#ifndef __CUDACC__
	Clear();
#endif
}

inline CUDA_HOST bool packed_data::HasData() const { return (unpackedLength > 0); }

inline CUDA_HOST CUDA_DEVICE bool packed_data::HasBitmap() const { return (bitmapLength > 0); }

}  // namespace himan

#endif /* HAVE_CUDA */
#endif /* PACKED_DATA_H */
