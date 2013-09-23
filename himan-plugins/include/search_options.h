/*
 * search_options.h
 *
 *  Created on: Dec 20, 2012
 *      Author: partio
 *
 * Simple struct to hold metadata that's needed to search data from
 * files (or cache).
 *
 */

#ifndef SEARCH_OPTIONS_H
#define SEARCH_OPTIONS_H

#include "forecast_time.h"
#include "param.h"
#include "level.h"
#include "plugin_configuration.h"

namespace himan
{
namespace plugin
{

struct search_options
{
    himan::forecast_time time;
    himan::param param;
    himan::level level;
    himan::producer prod;
    const std::shared_ptr<const himan::configuration> configuration;

	search_options(const himan::forecast_time& theTime,
					const himan::param& theParam,
					const himan::level& theLevel,
					const himan::producer& theProducer,
					std::shared_ptr<const himan::configuration> theConf)
	: time(theTime)
	, param(theParam)
	, level(theLevel)
	, prod(theProducer)
	, configuration(theConf)
	{}
};

} // namespace plugins
} // namespace himan

#endif /* SEARCH_OPTIONS_H */
