/**
 * File:   tpot_cuda.h
 * Author: partio
 *
 * Created on April 17, 2013, 3:32 PM
 * 
 * List of extern functions compiled by nvcc for plugins (compiled by gcc)
 */

#ifndef TPOT_CUDA_H
#define TPOT_CUDA_H

#ifdef HAVE_CUDA

#include "packed_data.h"

namespace himan
{
namespace plugin
{
namespace tpot_cuda
{

struct tpot_cuda_options
{
	const double* TIn;
	const double* PIn;
	double* TpOut;
	size_t N;
	bool isPackedData;
	unsigned short cudaDeviceIndex;
	int missingValuesCount;
	double TBase;
	double PScale;
	double PConst;
	bool isConstantPressure;
	
	simple_packed simplePackedT;
	simple_packed simplePackedP;

	tpot_cuda_options() : isPackedData(false), missingValuesCount(0) {}
};

void DoCuda(tpot_cuda_options& options);

} // namespace tpot_cuda
} // namespace plugin
} // namespace himan

#endif  /* HAVE_CUDA */
#endif	/* TPOT_CUDA_H */

