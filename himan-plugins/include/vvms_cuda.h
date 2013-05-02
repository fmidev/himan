/**
 * File:   vvms_cuda.h
 * Author: partio
 *
 * Created on February 17, 2013, 3:32 PMi
 * 
 * List of extern functions compiled by nvcc for plugins (compiled by gcc)
 */

#ifndef VVMS_CUDA_H
#define VVMS_CUDA_H
#ifdef HAVE_CUDA

#include "packed_data.h"

namespace himan
{
namespace plugin
{
namespace vvms_cuda
{

struct vvms_cuda_options
{
	size_t N;
	double PConst;
	double TBase;
	double PScale;
	bool isConstantPressure;
	bool pT;
	bool pVV;
	bool pP;
	unsigned short cudaDeviceIndex;
	int missingValuesCount;
	
	vvms_cuda_options() : isConstantPressure(false), pT(false), pVV(false), pP(false), missingValuesCount(0) {}
};

struct vvms_cuda_data
{
	double* T;
	double* P;
	double* VV;
	double* VVMS;

	simple_packed pT;
	simple_packed pVV;
	simple_packed pP;

	vvms_cuda_data() : T(0), P(0), VV(0), pT(), pVV(), pP() {}

};


void DoCuda(vvms_cuda_options& options, vvms_cuda_data& datas);

} // namespace vvms_cuda
} // namespace plugin
} // namespace himan

#endif	/* HAVE_CUDA */
#endif	/* VVMS_CUDA_H */

