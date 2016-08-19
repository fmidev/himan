/**
 * @file   interpolate.h
 * @author partio
 *
 * @date   July 5, 2016, 2:43 PM
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
}
}

#endif /* INTERPOLATE_H */
