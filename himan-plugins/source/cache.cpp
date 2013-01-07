/*
 * cache.cpp
 *
 *  Created on: Nov 21, 2012
 *      Author: partio
 */

#include "cache.h"
#include "logger_factory.h"

using namespace himan::plugin;

cache::cache()
{
    itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("cache"));
}
