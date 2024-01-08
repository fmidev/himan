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
	kFinished,
	kPending,
	kFailed,
	kSpilled
};

const std::unordered_map<HPWriteStatus, std::string> HPWriteStatusToString = {{HPWriteStatus::kUnknown, "unknown"},
                                                                              {HPWriteStatus::kFinished, "finished"},
                                                                              {HPWriteStatus::kPending, "pending"},
                                                                              {HPWriteStatus::kFailed, "failed"},
                                                                              {HPWriteStatus::kSpilled, "spilled"}};

namespace plugin
{
struct write_options
{
	std::shared_ptr<const plugin_configuration> configuration;
	bool use_bitmap;                                                  // use bitmap for grib if missing data exists
	bool write_empty_grid;                                            // write file even if all data is missing
	int precision;                                                    // precision (decimal points)
	std::vector<std::pair<std::string, std::string>> extra_metadata;  // additional metadata to be written to file

	write_options() : use_bitmap(true), write_empty_grid(true), precision(kHPMissingInt), extra_metadata()
	{
	}
};
}  // namespace plugin
}  // namespace himan

#endif /* WRITE_OPTIONS_H */
