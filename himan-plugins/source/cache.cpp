/*
 * cache.cpp
 *
 *  Created on: Nov 21, 2012
 *      Author: partio
 */

#include "cache.h"
#include "logger_factory.h"

using namespace hilpee::plugin;

cache::cache()
{
	itsLogger = logger_factory::Instance()->GetLog("cache");
}
