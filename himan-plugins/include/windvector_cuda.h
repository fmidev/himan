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

#include "simple_packed.h"

namespace himan
{
namespace plugin
{
namespace windvector_cuda
{

struct windvector_cuda_options
{
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
	int missingValuesCount;
	bool pU;
	bool pV;
	bool jScansPositive;

	windvector_cuda_options() 
		: targetType(kUnknownElement)
		, vectorCalculation(false)
		, needRotLatLonGridRotation(false)
		, missingValuesCount(0)
		, pU(false)
		, pV(false)
		, jScansPositive(true)
	{}

};

struct windvector_cuda_data
{
	double* u;
	double* v;
	double* speed;
	double* dir;
	double* vector;
	simple_packed* pU;
	simple_packed* pV;

	windvector_cuda_data() : u(0), v(0), speed(0), dir(0), vector(0), pU(0), pV(0) {}

};

void DoCuda(windvector_cuda_options& opts, windvector_cuda_data& data);


} // namespace windvector_cuda
} // namespace plugin
} // namespace himan

#endif  /* HAVE_CUDA */
#endif	/* WINDVECTOR_CUDA_H */
#endif  /* WINDVECTOR_HEADER_INCLUDE */