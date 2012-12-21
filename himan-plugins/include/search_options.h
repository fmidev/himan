/*
 * search_options.h
 *
 *  Created on: Dec 20, 2012
 *      Author: partio
 */

#ifndef SEARCH_OPTIONS_H
#define SEARCH_OPTIONS_H

#include "forecast_time.h"
#include "param.h"
#include "level.h"
#include "configuration.h"

namespace hilpee
{
namespace plugin
{

struct search_options {
	const hilpee::forecast_time& time;
	const hilpee::param& param;
	const hilpee::level& level;
	const hilpee::configuration& configuration;
};

} // namespace plugins
} // namespace hilpee

#endif /* SEARCH_OPTIONS_H */
