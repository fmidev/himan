#pragma once
#include "buffer.h"
#include "file_information.h"

namespace himan
{
namespace s3
{
enum class read_mode
{
	kSigned,    // always sign reads with credentials from environment
	kUnsigned  // never sign reads (anonymous access only)
};

buffer ReadFile(const file_information& fileInformation);
void WriteObject(const std::string& objectName, const himan::buffer& buff);
bool Exists(const std::string& objectName);
long unsigned int ObjectSize(const std::string& objectName);
void SetReadMode(read_mode mode);
read_mode GetReadMode();
}  // namespace s3
}  // namespace himan
