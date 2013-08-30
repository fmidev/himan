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

#ifdef HAVE_CUDA

#include "simple_packed.h"

namespace himan
{
namespace plugin
{
namespace tk2tc_cuda
{

struct tk2tc_cuda_options
{
	size_t N;
	bool pTK;
	unsigned short threadIndex;
	int missingValuesCount;
	
	tk2tc_cuda_options() : pTK(false), missingValuesCount(0) {}
};

struct tk2tc_cuda_data
{
	double* TK;
	double* TC;

	simple_packed* pTK;

	tk2tc_cuda_data() : pTK() {}
};

void DoCuda(tk2tc_cuda_options& options, tk2tc_cuda_data& datas);

} // namespace tk2tc_cuda
} // namespace plugin
} // namespace himan

#endif  /* HAVE_CUDA */
#endif	/* TK2TC_CUDA_H */

