/**
 * @file	write_options.h
 */

#ifndef WRITE_OPTIONS_H
#define WRITE_OPTIONS_H

#include <plugin_configuration.h>

namespace himan
{
enum class HPWriteStatus
{
	kUnknown = 0,
	kFinished,    // written to disk
	kPending,     // will be written to s3 later
	kFailed,      // failed to write to disk
	kSpilled,     // serialized to disk, will be written later to s3
	kDuplicated,  // duplicate of another file, nothing written
};

const std::unordered_map<HPWriteStatus, std::string> HPWriteStatusToString = {
    {HPWriteStatus::kUnknown, "unknown"}, {HPWriteStatus::kFinished, "finished"},
    {HPWriteStatus::kPending, "pending"}, {HPWriteStatus::kFailed, "failed"},
    {HPWriteStatus::kSpilled, "spilled"}, {HPWriteStatus::kDuplicated, "duplicated"}};

namespace plugin
{
struct write_options
{
	std::shared_ptr<const plugin_configuration> configuration;
	bool use_bitmap;                                                  // use bitmap for grib if missing data exists
	bool write_empty_grid;                                            // write file even if all data is missing
	int precision;                                                    // precision (decimal points)
	std::vector<std::pair<std::string, std::string>> extra_metadata;  // additional metadata to be written to file
	bool replace_cache;                                               // replace existing data in cache if it exists

	write_options()
	    : use_bitmap(true), write_empty_grid(true), precision(kHPMissingInt), extra_metadata(), replace_cache(false)
	{
	}
};
}  // namespace plugin
}  // namespace himan

#endif /* WRITE_OPTIONS_H */
