/**
 * @file packed_data.cpp
 *
 * @date Apr 11, 2013
 * @author partio
 */

#include "packed_data.h"

#ifdef HAVE_CUDA

using namespace himan;

void packed_data::Set(unsigned char* newData, size_t newDataLength)
{
	if (data)
	{
		cudaFreeHost(data);
	}

	dataLength = newDataLength;
	data = newData;
}

void packed_data::Resize(size_t newDataLength)
{
	assert(newDataLength > dataLength);

	dataLength = newDataLength;
	unsigned char* newData = 0;

	cudaMallocHost(reinterpret_cast<void**> (&newData), dataLength * sizeof(unsigned char));

	memcpy(newData, data, dataLength * sizeof(unsigned char));

	cudaFreeHost(data);

	data = newData;
	
}

#endif
