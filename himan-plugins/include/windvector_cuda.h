/* 
 * File:   windvector_cuda.h
 * Author: partio
 *
 * Created on March 14, 2013, 2:17 PM
 */

#ifndef WINDVECTOR_CUDA_H
#define	WINDVECTOR_CUDA_H

#include "packed_data.h"

namespace himan
{
namespace plugin
{
namespace windvector_cuda
{

struct windvector_cuda_options
{
	double* Uin;
	double* Vin;
	double* dataOut;
	size_t sizeX;
	size_t sizeY;
	double firstLatitude;
	double firstLongitude;
	double di;
	double dj;
	double southPoleLat;
	double southPoleLon;
	bool vectorCalculation;
	bool dirCalculation;
	bool needRotLatLonGridRotation;
	unsigned short cudaDeviceIndex;
	bool isPackedData;
	int missingValuesCount;

	simple_packed simplePackedU, simplePackedV;

	windvector_cuda_options() : vectorCalculation(false), dirCalculation(true), needRotLatLonGridRotation(false), isPackedData(false), missingValuesCount(0) {}

};

void DoCuda(windvector_cuda_options& opts);


} // namespace windvector_cuda
} // namespace plugin
} // namespace himan

#endif	/* WINDVECTOR_CUDA_H */

