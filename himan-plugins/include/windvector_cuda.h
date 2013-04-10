/* 
 * File:   windvector_cuda.h
 * Author: partio
 *
 * Created on March 14, 2013, 2:17 PM
 */

#ifndef WINDVECTOR_CUDA_H
#define	WINDVECTOR_CUDA_H

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
	unsigned short CudaDeviceIndex;

	windvector_cuda_options() : vectorCalculation(false), dirCalculation(true), needRotLatLonGridRotation(false) {}

};

void DoCuda(windvector_cuda_options& opts);


} // namespace windvector_cuda
} // namespace plugin
} // namespace himan

#endif	/* WINDVECTOR_CUDA_H */

