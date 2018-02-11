#pragma once
#include "buffer.h"
#include "file_information.h"

namespace himan
{
namespace s3
{
buffer ReadFile(const file_information& fileInformation);
}  // namespace s3
}  // namespace himan
