/* 
 * File:   windvector_cuda.h
 * Author: partio
 *
 * Created on March 14, 2013, 2:17 PM
 */

#if defined WINDVECTOR_HEADER_INCLUDE || defined __CUDACC__ // don't want to pollute whole namespace with this s*it

namespace himan
{
namespace plugin
{

enum HPTargetType
{
	kUnknownElement = 0,
	kWind,
	kGust,
	kSea,
	kIce
};

} // namespace plugin
} // namespace himan


#endif

#ifndef WINDVECTOR_HEADER_INCLUDE
#ifndef WINDVECTOR_CUDA_H
#define	WINDVECTOR_CUDA_H

#ifdef HAVE_CUDA

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
	HPTargetType targetType;
	bool vectorCalculation;
	bool needRotLatLonGridRotation;
	unsigned short cudaDeviceIndex;
	bool isPackedData;
	int missingValuesCount;

	//simple_packed simplePackedU, simplePackedV;

	windvector_cuda_options() : targetType(kUnknownElement), vectorCalculation(false), needRotLatLonGridRotation(false), isPackedData(false), missingValuesCount(0) {}

};

void DoCuda(windvector_cuda_options& opts);


} // namespace windvector_cuda
} // namespace plugin
} // namespace himan

#endif  /* HAVE_CUDA */
#endif	/* WINDVECTOR_CUDA_H */
#endif  /* WINDVECTOR_HEADER_INCLUDE */