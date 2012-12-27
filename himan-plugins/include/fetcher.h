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
#include "configuration.h"
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

		std::shared_ptr<info> Fetch(std::shared_ptr<const configuration> theConfiguration, std::shared_ptr<const forecast_time> theValidTime, std::shared_ptr<const level> theLevel, std::shared_ptr<const param> theParam);

	private:

		std::vector<std::shared_ptr<info>> FromFile(const std::string& theFileName, const search_options& options, bool theReadContents = true);
		std::vector<std::shared_ptr<info>> FromGrib(const std::string& theInputFile, const search_options& options, bool theReadContents = true);
		std::vector<std::shared_ptr<info>> FromQueryData(const std::string& theInputFile, const search_options& options, bool theReadContents = true);

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
