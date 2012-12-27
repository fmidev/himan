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
#include "configuration.h"

namespace himan
{
namespace plugin
{

// Could use reference to const shared_ptr to avoid incrementing shared_ptr counter
// but is that really worth it

struct search_options
{
	std::shared_ptr<const himan::forecast_time> time;
	std::shared_ptr<const himan::param> param;
	std::shared_ptr<const himan::level> level;
	std::shared_ptr<const himan::configuration> configuration;
};

} // namespace plugins
} // namespace himan

#endif /* SEARCH_OPTIONS_H */
