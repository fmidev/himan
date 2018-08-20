/**
 * @file interpolator.cpp
 *
 */

#include "interpolator.h"
#include "area_interpolation.h"
#include "plugin_factory.h"

using namespace std;
using namespace himan::plugin;

typedef lock_guard<mutex> Lock;

std::map<std::string, himan::interpolate::area_interpolation> interpolator::cache;

bool interpolator::Insert(grid* source, grid* target)
{
	std::pair<std::string,himan::interpolate::area_interpolation> insertValue;

	try
	{
        	insertValue.first = source->Identifier() + target->Identifier();
		insertValue.second = himan::interpolate::area_interpolation(source,target);
	}
	catch (const std::exception& e)
	{
		return false;
	}

	return cache.insert(std::move(insertValue)).second;
}

bool interpolator::Interpolate(grid* source, grid* target)
{
	auto it = cache.find(source->Identifier()+target->Identifier());

	if (it != cache.end())
	{
		it->second.Interpolate(source,target);
		return true;
	}
	return false;
}

