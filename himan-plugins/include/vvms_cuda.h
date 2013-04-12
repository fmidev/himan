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
	const double* TIn;
	const double* PIn;
	const double* VVIn;
	double* VVOut;
	size_t N;
	double PConst;
	double TBase;
	double PScale;
	bool isConstantPressure;
	bool isPackedData;
	unsigned short cudaDeviceIndex;
	int missingValuesCount;
	
	simple_packed simplePackedT;
	simple_packed simplePackedVV;
	simple_packed simplePackedP;

	vvms_cuda_options() : isConstantPressure(false), isPackedData(false), missingValuesCount(0) {}
};

void DoCuda(vvms_cuda_options& options);

} // namespace vvms_cuda
} // namespace plugin
} // namespace himan

#endif	/* HAVE_CUDA */
#endif	/* VVMS_CUDA_H */

