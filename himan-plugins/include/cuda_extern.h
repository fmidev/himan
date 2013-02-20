/**
 * File:   cuda_extern.h
 * Author: partio
 *
 * Created on February 17, 2013, 3:32 PM
 */

#ifndef CUDA_EXTERN_H
#define	CUDA_EXTERN_H

#ifdef HAVE_CUDA
namespace himan
{
namespace plugin
{

namespace tk2tc_cuda
{
void DoCuda(const float* Tin, float* Tout, size_t N, unsigned short deviceIndex);
}

namespace tpot_cuda
{
void DoCuda(const float* Tin, float TBase, const float* Pin, float TScale, float* TPout, size_t N, float PConst, unsigned short index);
}

namespace vvms_cuda
{
void DoCuda(const float* Tin, float TBase, const float* Pin, float PScale, const float* VVin, float* VVout, size_t N, float PConst, unsigned short deviceIndex);
}

}
}
#endif /* HAVE_CUDA */

#endif	/* CUDA_EXTERN_H */

