#pragma once

#include "himan_common.h"
#include <optional>
#include <string>

namespace himan
{
struct file_information
{
	std::string file_location;                // /path/to/file
	std::string file_server;                  // server/host where file is accessible
	HPFileType file_type;                     // GRIB,csv, etc see himan_common.h
	HPFileStorageType storage_type;           // POSIX filesystem, S3, ...
	std::optional<unsigned long> message_no;  // "message" ordinal number
	std::optional<unsigned long> offset;      // "message" offset from file beginning
	std::optional<unsigned long> length;      // "message" length
};
}  // namespace himan
