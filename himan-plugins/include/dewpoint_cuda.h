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

#include "packed_data.h"

namespace himan
{
namespace plugin
{
namespace dewpoint_cuda
{

struct dewpoint_cuda_options
{
	const double* TIn;
	const double* RHIn;
	double* TDOut;
	size_t N;
	bool isPackedData;
	unsigned short cudaDeviceIndex;
	int missingValuesCount;
	double TBase;
	
	simple_packed simplePackedT;
	simple_packed simplePackedRH;

	dewpoint_cuda_options() : isPackedData(false), missingValuesCount(0) {}
};

void DoCuda(dewpoint_cuda_options& options);

} // namespace dewpoint_cuda
} // namespace plugin
} // namespace himan

#endif  /* HAVE_CUDA */
#endif	/* DEWPOINT_CUDA_H */

