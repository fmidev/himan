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

bool InterpolateArea(info& base, std::vector<info_t> infos, bool useCudaForInterpolation = true);

bool Interpolate(info& base, std::vector<info_t>& infos, bool useCudaForInterpolation = true);

bool IsVectorComponent(const std::string& paramName);

HPInterpolationMethod InterpolationMethod(const std::string& paramName, HPInterpolationMethod interpolationMethod);

void RotateVectorComponents(info& UInfo, info& VInfo, bool useCuda);

void RotateVectorComponentsCPU(info& UInfo, info& VInfo);
}
}

#endif /* INTERPOLATE_H */
