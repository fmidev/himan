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

struct search_options
{
    const himan::forecast_time& time;
    const himan::param& param;
    const himan::level& level;
    const std::shared_ptr<const himan::configuration> configuration;
};

} // namespace plugins
} // namespace himan

#endif /* SEARCH_OPTIONS_H */
