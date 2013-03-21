/* 
 * File:   windvector_cuda_options.h
 * Author: partio
 *
 * Created on March 14, 2013, 2:17 PM
 */

#ifndef WINDVECTOR_CUDA_OPTIONS_H
#define	WINDVECTOR_CUDA_OPTIONS_H

struct windvector_cuda_options
{
	float* Uin;
	float* Vin;
	float* dataOut;
	size_t sizeX;
	size_t sizeY;
	float firstLatitude;
	float firstLongitude;
	float di;
	float dj;
	float southPoleLat;
	float southPoleLon;
	bool vectorCalculation;
	bool dirCalculation;
	bool needRotLatLonGridRotation;
	unsigned short CudaDeviceIndex;

};

#endif	/* WINDVECTOR_CUDA_OPTIONS_H */

