/**
 * @file cache.cpp
 *
 */

#include "cache.h"
#include "info.h"
#include "logger.h"
#include "plugin_factory.h"
#include <boost/lexical_cast.hpp>
#include <time.h>

using namespace std;
using namespace himan::plugin;

typedef lock_guard<mutex> Lock;

cache::cache()
{
	itsLogger = logger("cache");
}
string cache::UniqueName(const info& info)
{
	stringstream ss;

	// clang-format off

	ss << info.Producer().Id() << "_"
	   << info.Time().OriginDateTime().String("%Y-%m-%d %H:%M:%S") << "_"
	   << info.Time().ValidDateTime().String("%Y-%m-%d %H:%M:%S") << "_"
	   << info.Param().Name() << "_"
	   << static_cast<string>(info.Level()) << "_"
	   << info.ForecastType().Type() << "_"
	   << info.ForecastType().Value();

	// clang-format on

	return ss.str();
}

string cache::UniqueNameFromOptions(search_options& options)
{
	ASSERT(options.configuration->DatabaseType() == kNoDatabase || options.prod.Id() != kHPMissingInt);
	stringstream ss;

	// clang-format off

	ss << options.prod.Id() << "_"
	   << options.time.OriginDateTime().String("%Y-%m-%d %H:%M:%S") << "_"
	   << options.time.ValidDateTime().String("%Y-%m-%d %H:%M:%S") << "_"
	   << options.param.Name() << "_"
	   << static_cast<string>(options.level) << "_"
	   << options.ftype.Type() << "_"
	   << options.ftype.Value();

	// clang-format on

	return ss.str();
}

void cache::Insert(info_t anInfo, bool pin)
{
	SplitToPool(anInfo, pin);
}
void cache::SplitToPool(info_t anInfo, bool pin)
{
	auto localInfo = make_shared<info>(*anInfo);

	// Cached data is never replaced by another data that has
	// the same uniqueName

	const string uniqueName = UniqueName(*localInfo);

	if (cache_pool::Instance()->Find(uniqueName))
	{
		itsLogger.Trace("Data with key " + uniqueName + " already exists at cache");

		// Update timestamp of this cache item
		cache_pool::Instance()->UpdateTime(uniqueName);
		return;
	}

#ifdef HAVE_CUDA
	if (localInfo->Grid()->IsPackedData())
	{
		itsLogger.Trace("Removing packed data from cached info");
		localInfo->Grid()->PackedData().Clear();
	}
#endif

	ASSERT(!localInfo->Grid()->IsPackedData());

	// localInfo might contain multiple grids. When adding data to cache, we need
	// to make sure that single info contains only single grid.

	if (localInfo->DimensionSize() > 1)
	{
		auto newInfo = make_shared<info>(localInfo->ForecastType(), localInfo->Time(), localInfo->Level(), localInfo->Param());
		newInfo->Grid(localInfo->SharedGrid());
		localInfo = newInfo;
	}

	ASSERT(localInfo->DimensionSize() == 1);
	cache_pool::Instance()->Insert(uniqueName, localInfo, pin);
}

vector<shared_ptr<himan::info>> cache::GetInfo(search_options& options)
{
	string uniqueName = UniqueNameFromOptions(options);

	vector<shared_ptr<himan::info>> info;

	if (cache_pool::Instance()->Find(uniqueName))
	{
		info.push_back(cache_pool::Instance()->GetInfo(uniqueName));
		itsLogger.Trace("Found matching data for " + uniqueName);
	}

	return info;
}

void cache::Clean()
{
	cache_pool::Instance()->Clean();
}
size_t cache::Size() const
{
	return cache_pool::Instance()->Size();
}
cache_pool* cache_pool::itsInstance = NULL;

cache_pool::cache_pool() : itsCacheLimit(-1)
{
	itsLogger = logger("cache_pool");
}

cache_pool* cache_pool::Instance()
{
	if (!itsInstance)
	{
		itsInstance = new cache_pool();
	}

	return itsInstance;
}

void cache_pool::CacheLimit(int theCacheLimit)
{
	itsCacheLimit = theCacheLimit;
}
bool cache_pool::Find(const string& uniqueName)
{
	Lock lock(itsAccessMutex);
	return itsCache.count(uniqueName) > 0;
}
void cache_pool::Insert(const string& uniqueName, shared_ptr<himan::info> anInfo, bool pin)
{
	cache_item item;
	item.info = anInfo;
	item.access_time = time(nullptr);
	item.pinned = pin;

	{
		Lock lock(itsAccessMutex);

		itsCache.insert(pair<string, cache_item>(uniqueName, item));
	}

	itsLogger.Trace("Data added to cache with name: " + uniqueName + ", pinned: " + to_string(pin));

	if (itsCacheLimit > -1 && itsCache.size() > static_cast<size_t>(itsCacheLimit))
	{
		Clean();
	}
}

void cache_pool::UpdateTime(const std::string& uniqueName)
{
	Lock lock(itsAccessMutex);
	itsCache[uniqueName].access_time = time(nullptr);
}
void cache_pool::Clean()
{
	ASSERT(itsCacheLimit > 0);
	if (itsCache.size() <= static_cast<size_t>(itsCacheLimit))
	{
		return;
	}

	string oldestName;
	time_t oldestTime = INT_MAX;

	{
		Lock lock(itsAccessMutex);

		for (const auto& kv : itsCache)
		{
			if (kv.second.access_time < oldestTime && !kv.second.pinned)
			{
				oldestName = kv.first;
				oldestTime = kv.second.access_time;
			}
		}

		ASSERT(!oldestName.empty());

		itsCache.erase(oldestName);
	}

	itsLogger.Trace("Data cleared from cache: " + oldestName + " with time: " + to_string(oldestTime));
	itsLogger.Trace("Cache size: " + to_string(itsCache.size()));
}

shared_ptr<himan::info> cache_pool::GetInfo(const string& uniqueName)
{
	Lock lock(itsAccessMutex);

	return make_shared<info>(*itsCache[uniqueName].info);
}

size_t cache_pool::Size() const
{
	return itsCache.size();
}
