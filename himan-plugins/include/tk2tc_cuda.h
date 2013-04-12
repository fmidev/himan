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

#include "simple_packing_options.h"

namespace himan
{
namespace plugin
{
namespace tk2tc_cuda
{

struct tk2tc_cuda_options
{
	const double* TIn;
	const unsigned char* TInPacked;
	double* TOut;
	size_t N;
	size_t NPacked;
	bool isPackedData;
	unsigned short cudaDeviceIndex;
	int missingValuesCount;
	struct simple_packing_options simple_packing;

	tk2tc_cuda_options() : isPackedData(false), simple_packing() {}
};

void DoCuda(tk2tc_cuda_options& options);

} // namespace tk2tc_cuda
} // namespace plugin
} // namespace himan

#endif	/* TK2TC_CUDA_H */

