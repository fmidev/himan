/**
 * File:   tk2tc_cuda.h
 * Author: partio
 *
 * Created on February 17, 2013, 3:32 PM
 * 
 * List of extern functions compiled by nvcc for plugins (compiled by gcc)
 */

#ifndef TK2TC_CUDA_H
#define TK2TC_CUDA_H

#include "packed_data.h"

namespace himan
{
namespace plugin
{
namespace tk2tc_cuda
{

struct tk2tc_cuda_options
{
	const double* TIn;
	double* TOut;
	size_t N;
	bool isPackedData;
	unsigned short cudaDeviceIndex;
	int missingValuesCount;
	
	simple_packed simplePackedT;

	tk2tc_cuda_options() : isPackedData(false), missingValuesCount(0) {}
};

void DoCuda(tk2tc_cuda_options& options);

} // namespace tk2tc_cuda
} // namespace plugin
} // namespace himan

#endif	/* TK2TC_CUDA_H */

