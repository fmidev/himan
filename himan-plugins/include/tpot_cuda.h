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
	size_t N;
	bool pT;
	bool pP;
	unsigned short cudaDeviceIndex;
	int missingValuesCount;
	double TBase;
	double PScale;
	double PConst;
	bool isConstantPressure;
	
	tpot_cuda_options() : pT(false), pP(false), missingValuesCount(0) {}
};

struct tpot_cuda_data
{
	double* T;
	double* P;
	double* Tp;

	simple_packed pT;
	simple_packed pP;

	tpot_cuda_data() : T(0), P(0), Tp(0), pT(), pP() {}
};

void DoCuda(tpot_cuda_options& options, tpot_cuda_data& datas);

} // namespace tpot_cuda
} // namespace plugin
} // namespace himan

#endif  /* HAVE_CUDA */
#endif	/* TPOT_CUDA_H */

