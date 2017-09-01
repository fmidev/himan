/**
 * @file packed_data.cpp
 *
 */

#include "packed_data.h"

#ifdef HAVE_CUDA

using namespace himan;

void packed_data::Set(unsigned char* newData, size_t newPackedLength, size_t newUnpackedLength)
{
	if (packedLength)
	{
		assert(data);
		CUDA_CHECK(cudaFreeHost(data));
	}

	packedLength = newPackedLength;
	unpackedLength = newUnpackedLength;
	data = newData;
}

void packed_data::Bitmap(int* newBitmap, size_t newBitmapLength)
{
	if (newBitmapLength && bitmapLength)
	{
		assert(bitmap);
		CUDA_CHECK(cudaFreeHost(bitmap));
	}

	bitmapLength = newBitmapLength;
	bitmap = newBitmap;
}

void packed_data::Resize(size_t newPackedLength, size_t newUnpackedLength)
{
	assert(newPackedLength > packedLength);

	packedLength = newPackedLength;
	unpackedLength = newUnpackedLength;

	unsigned char* newData = 0;

	CUDA_CHECK(
	    cudaHostAlloc(reinterpret_cast<void**>(&newData), packedLength * sizeof(unsigned char), cudaHostAllocMapped));

	memcpy(newData, data, packedLength * sizeof(unsigned char));

	CUDA_CHECK(cudaFreeHost(data));

	data = newData;
}

packed_data::packed_data(const packed_data& other)
    : packedLength(other.packedLength),
      unpackedLength(other.unpackedLength),
      bitmapLength(other.bitmapLength),
      packingType(other.packingType)
{
	data = 0;
	bitmap = 0;

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
	if (packedLength)
	{
		assert(data);
		CUDA_CHECK(cudaFreeHost(data));
		packedLength = 0;
		data = 0;
	}

	if (bitmapLength)
	{
		assert(bitmap);
		CUDA_CHECK(cudaFreeHost(bitmap));
		bitmapLength = 0;
		bitmap = 0;
	}

	unpackedLength = 0;
}

#endif
