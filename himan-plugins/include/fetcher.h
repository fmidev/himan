/*
 * fetcher.h
 *
 *  Created on: Nov 21, 2012
 *      Author: partio
 *
 * Class purpose is to server as an intermediator between a
 * compiled plugin asking for data and the different plugins
 * serving the different aspects of the data.
 *
 * In other words when a plugin asks for data for a parameter,
 * this plugin will return the data.
 *
 * 1) check cache if the data exists there; if so return it
 * 2) check auxiliary files specified at command line for data
 * 3) check neons for data
 * 4) fetch data if found
 * 5) store data to cache
 * 6) return data to caller
 *
 */

#ifndef FETCHER_H
#define FETCHER_H

#include "auxiliary_plugin.h"
#include "plugin_configuration.h"
#include "search_options.h"

namespace himan
{
namespace plugin
{

class fetcher : public auxiliary_plugin
{
public:
    fetcher();

    virtual ~fetcher() {}

    fetcher(const fetcher& other) = delete;
    fetcher& operator=(const fetcher& other) = delete;

    virtual std::string ClassName() const
    {
        return "himan::plugin::fetcher";
    }

    virtual HPPluginClass PluginClass() const
    {
        return kAuxiliary;
    }

    virtual HPVersionNumber Version() const
    {
        return HPVersionNumber(0, 1);
    }

	/**
	 *
     * @param config
     * @param requestedValidTime
     * @param requestedLevel
     * @param requestedParam
	 * @param readPackedData Whether to read unpacked data (from grib only!). Caller must do unpacking.
     * @return
     */
    std::shared_ptr<info> Fetch(std::shared_ptr<const plugin_configuration> config, const forecast_time& requestedValidTime, const level& requestedLevel, const param& requestedParam, bool readPackedData = false);

private:

    std::vector<std::shared_ptr<info>> FromCache(const search_options& options);

    std::vector<std::shared_ptr<info>> FromFile(const std::vector<std::string>& files,
												const search_options& options,
												bool readContents = true,
												bool readPackedData = false);

    /**
     * @brief Return all data from a querydata file, overcoat for himan::plugin::querydata::FromFile().
     * @see himan::plugin::grib::FromFile()
     *
     * @param inputFile Input file name
     * @param options Search options (param, level, time)
     * @param readContents Specify if data should also be read (and not only metadata)
	 * @param readPackedData Whether to read packed data. Caller must do unpacking.
     *
     * @return A vector of shared_ptr'd infos.
     */

    std::vector<std::shared_ptr<info>> FromGrib(const std::string& inputFile, const search_options& options, bool readContents = true, bool readPackedData = false);
	
    /**
     * @brief Return all data from a querydata file, overcoat for himan::plugin::querydata::FromFile().
     * @see himan::plugin::querydata::FromFile()
     *
     * @param inputFile Input file name
     * @param options Search options (param, level, time)
     * @param readContents Specify if data should also be read (and not only metadata)
     *
     * @return A vector of shared_ptr'd infos. Vector size is always 0 or 1.
     */

    std::vector<std::shared_ptr<info>> FromQueryData(const std::string& inputFile, const search_options& options, bool readContents = true);

    HPFileType FileType(const std::string& theInputFile);

};

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<fetcher> (new fetcher());
}

#endif /* HIMAN_AUXILIARY_INCLUDE */

} // namespace plugin
} // namespace himan

#endif /* FETCHER_H */
