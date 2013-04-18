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

} // namespace plugin
} // namespace himan

#endif	/* CUDA_EXTERN_H */
