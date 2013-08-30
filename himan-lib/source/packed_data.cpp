/**
 * @file packed_data.cpp
 *
 * @date Apr 11, 2013
 * @author partio
 */

#include "packed_data.h"

#ifdef HAVE_CUDA

using namespace himan;

void packed_data::Set(unsigned char* newData, size_t newPackedLength, size_t newUnpackedLength)
{
	if (packedLength)
	{
		cudaFreeHost(data);
	}

	packedLength = newPackedLength;
	unpackedLength = newUnpackedLength;
	data = newData;
}

void packed_data::Bitmap(int* newBitmap, size_t newBitmapLength)
{
	if (newBitmapLength)
	{
		cudaFreeHost(bitmap);
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

	cudaHostAlloc(reinterpret_cast<void**> (&newData), packedLength * sizeof(unsigned char), cudaHostAllocMapped);

	memcpy(newData, data, packedLength * sizeof(unsigned char));

	cudaFreeHost(data);

	data = newData;
	
}

#endif
