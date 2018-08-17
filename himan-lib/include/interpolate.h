/**
 * @file   interpolate.h
 *
 */

#ifndef INTERPOLATE_H
#define INTERPOLATE_H

#include "plugin_configuration.h"

namespace himan
{
namespace interpolate
{
bool InterpolateAreaCPU(info& base, info& source, matrix<double>& targetData);

bool InterpolateArea(info& base, info_t source, bool useCudaForInterpolation = true);

bool Interpolate(info& base, std::vector<info_t>& infos, bool useCudaForInterpolation = true);

bool IsVectorComponent(const std::string& paramName);

HPInterpolationMethod InterpolationMethod(const std::string& paramName, HPInterpolationMethod interpolationMethod);

void RotateVectorComponents(info& UInfo, info& VInfo, bool useCuda);

void RotateVectorComponentsCPU(info& UInfo, info& VInfo);

#ifdef HAVE_CUDA
bool InterpolateAreaGPU(himan::info& targetInfo, himan::info& baseInfo, himan::matrix<double>& targetData);
void RotateVectorComponentsGPU(info& UInfo, info& VInfo, cudaStream_t& stream, double* d_u, double* d_v);
#endif

bool IsSupportedGridForRotation(HPGridType type);
}
}

#endif /* INTERPOLATE_H */
