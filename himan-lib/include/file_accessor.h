#pragma once

#include "buffer.h"
#include "file_information.h"

namespace himan
{
class file_accessor
{
   public:
	file_accessor() = default;
	~file_accessor() = default;

	buffer Read(const file_information& finfo) const;
};
}  // namespace himan
