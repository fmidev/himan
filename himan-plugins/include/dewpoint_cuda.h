/**
 * File:   dewpoint_cuda.h
 * Author: partio
 *
 * Created on February 17, 2013, 3:32 PM
 * 
 * List of extern functions compiled by nvcc for plugins (compiled by gcc)
 */

#ifndef DEWPOINT_CUDA_H
#define DEWPOINT_CUDA_H

#ifdef HAVE_CUDA

#include "simple_packed.h"

namespace himan
{
namespace plugin
{
namespace dewpoint_cuda
{

struct dewpoint_cuda_options
{
	size_t N;
	unsigned short cudaDeviceIndex;
	size_t missingValuesCount;
	double TBase;
	bool pT;
	bool pRH;
	
	dewpoint_cuda_options() : missingValuesCount(0), pT(false), pRH(false) {}
};

struct dewpoint_cuda_data
{
	double* T;
	double* RH;
	double* TD;
	simple_packed* pT;
	simple_packed* pRH;

	dewpoint_cuda_data() : T(0), RH(0), TD(0), pT(0), pRH(0) {}

};

void DoCuda(dewpoint_cuda_options& options, dewpoint_cuda_data& datas);

} // namespace dewpoint_cuda
} // namespace plugin
} // namespace himan

#endif  /* HAVE_CUDA */
#endif	/* DEWPOINT_CUDA_H */

