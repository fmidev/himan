/**
 * @file packed_data.cpp
 *
 */

#include "packed_data.h"

#ifdef HAVE_CUDA

using namespace himan;

void packed_data::Set(unsigned char* newData, size_t newPackedLength, size_t newUnpackedLength)
{
	if (data)
	{
		CUDA_CHECK(cudaFreeHost(data));
	}

	packedLength = newPackedLength;
	unpackedLength = newUnpackedLength;
	data = newData;
}

void packed_data::Bitmap(int* newBitmap, size_t newBitmapLength)
{
	if (bitmap)
	{
		CUDA_CHECK(cudaFreeHost(bitmap));
	}

	bitmapLength = newBitmapLength;
	bitmap = newBitmap;
}

void packed_data::Resize(size_t newPackedLength, size_t newUnpackedLength)
{
	ASSERT(newPackedLength > packedLength);

	unsigned char* newData = nullptr;

	CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&newData), newPackedLength * sizeof(unsigned char),
	                         cudaHostAllocMapped));

	memcpy(newData, data, packedLength * sizeof(unsigned char));

	packedLength = newPackedLength;
	unpackedLength = newUnpackedLength;

	CUDA_CHECK(cudaFreeHost(data));

	data = newData;
}

packed_data::packed_data(const packed_data& other)
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

void packed_data::Clear()
{
	if (data)
	{
		CUDA_CHECK(cudaFreeHost(data));
		packedLength = 0;
		data = nullptr;
	}

	if (bitmap)
	{
		CUDA_CHECK(cudaFreeHost(bitmap));
		bitmapLength = 0;
		bitmap = nullptr;
	}

	unpackedLength = 0;
}

#endif
