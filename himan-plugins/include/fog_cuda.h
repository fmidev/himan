/**
 * File:   fog_cuda.h
 * Author: partio, peramaki
 *
 * Created on August 1, 2013, 11:48 AM
 * 
 * List of extern functions compiled by nvcc for plugins (compiled by gcc)
 */

#ifndef FOG_CUDA_H
#define FOG_CUDA_H

#ifdef HAVE_CUDA

#include "packed_data.h"

namespace himan
{
namespace plugin
{
namespace fog_cuda
{

struct fog_cuda_options
{
	size_t N;
	unsigned short cudaDeviceIndex;
	int missingValuesCount;
	bool pDTC2M;
	bool pTKGround;
	bool pFF10M;
	
	fog_cuda_options() : missingValuesCount(0), pDTC2M(false), pTKGround(false), pFF10M(false) {}
};

struct fog_cuda_data
{
	double* DTC2M;
	double* TKGround;
	double* FF10M;
	double* F;
	simple_packed pDTC2M;
	simple_packed pTKGround;
	simple_packed pFF10M;

	fog_cuda_data() : DTC2M(0), TKGround(0), FF10M(0), pDTC2M(), pTKGround(), pFF10M() {}

};

void DoCuda(fog_cuda_options& options, fog_cuda_data& datas);

} // namespace fog_cuda
} // namespace plugin
} // namespace himan

#endif  /* HAVE_CUDA */
#endif	/* FOG_CUDA_H */

