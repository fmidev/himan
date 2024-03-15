#pragma once

/**
 * @file filename.h
 *
 *
 * @brief Namespace containing functions to create correct filenames
 */

#include "plugin_configuration.h"
#include "info.h"

namespace himan
{
namespace util
{
namespace filename
{

template <typename T>
std::string MakeFileName(const info<T>& info, const plugin_configuration& conf);

}  // namespace filename
}  // namespace util
}  // namespace himan
