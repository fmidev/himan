/**
 * @file   si_cuda.h
 * @author partio
 *
 * @date March 14, 2013, 2:17 PM
 */

#pragma once

#ifdef HAVE_CUDA
#include "info.h"
#include "info_simple.h"
#include "cuda_helper.h"
#include "plugin_configuration.h"

namespace himan
{
namespace plugin
{
namespace si_cuda 
{

std::pair<std::vector<double>,std::vector<double>> GetHighestThetaETAndTDGPU(const std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<info> myTargetInfo);
std::pair<std::vector<double>,std::vector<double>> GetLFCGPU(const std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<info> myTargetInfo, std::vector<double>& T, std::vector<double>& P, std::vector<double>& TenvLCL);


} // namespace si_cuda
} // namespace plugin
} // namespace himan

#endif  /* HAVE_CUDA */
