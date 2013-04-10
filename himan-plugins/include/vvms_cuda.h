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
	const unsigned char* TInPacked;
	const unsigned char* PInPacked;
	const unsigned char* VVInPacked;
	double* VVOut;
	size_t N;
	size_t NPacked;
	double PConst;
	double TBase;
	double PScale;
	bool isConstantPressure;
	bool isPackedData;
	unsigned short cudaDeviceIndex;
	long bitsPerValue;
	double binaryScaleFactor;
	double decimalScaleFactor;
	double referenceValue;

	vvms_cuda_options() : isConstantPressure(false), isPackedData(false) {}
};

void DoCuda(vvms_cuda_options& options);

} // namespace vvms_cuda
} // namespace plugin
} // namespace himan

#endif	/* VVMS_CUDA_H */

