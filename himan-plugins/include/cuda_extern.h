/**
 * File:   cuda_extern.h
 * Author: partio
 *
 * Created on February 17, 2013, 3:32 PMi
 * 
 * List of extern functions compiled by nvcc for plugins (compiled by gcc)
 */

#ifndef CUDA_EXTERN_H
#define CUDA_EXTERN_H

namespace himan
{
namespace plugin
{

namespace tk2tc_cuda
{
void DoCuda(const double* Tin, double* Tout, size_t N, unsigned short deviceIndex);
}

namespace tpot_cuda
{
void DoCuda(const double* Tin, double TBase, const double* Pin, double TScale, double* TPout, size_t N, double PConst, unsigned short index);
}

} // namespace plugin
} // namespace himan

#endif	/* CUDA_EXTERN_H */
