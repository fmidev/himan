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

#include "info_simple.h"

namespace himan
{
namespace plugin
{
namespace tk2tc_cuda
{

struct options
{
	size_t N;
	size_t missing;
	double scale;
	double base;

	info_simple* source;
	info_simple* dest;

	options() : N(0), missing(0), scale(1), base(0), source(0), dest(0) {}
};

void Process(options& opts);

__global__ void Calculate(const double* __restrict__ d_source, double* __restrict__ d_dest, options opts, int* d_missing);


} // namespace tk2tc_cuda
} // namespace plugin
} // namespace himan

#endif  /* HAVE_CUDA */
#endif	/* TK2TC_CUDA_H */

