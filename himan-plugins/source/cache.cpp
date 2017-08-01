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

std::mutex itsCheckMutex;

cache::cache() { itsLogger = logger("cache"); }
string cache::UniqueName(const info& info)
{
	string producer_id = boost::lexical_cast<string>(info.Producer().Id());
	string forecast_time = info.Time().OriginDateTime().String("%Y-%m-%d_%H:%M:%S");
	string valid_time = info.Time().ValidDateTime().String("%Y-%m-%d_%H:%M:%S");
	string param = info.Param().Name();
	string level_value = boost::lexical_cast<string>(info.Level().Value());
	string level = HPLevelTypeToString.at(info.Level().Type());
	string forecast_type = boost::lexical_cast<string>(info.ForecastType().Type());
	string forecast_type_value = boost::lexical_cast<string>(info.ForecastType().Value());
	return producer_id + '_' + forecast_time + '_' + valid_time + '_' + param + '_' + level + '_' + level_value + '_' +
	       forecast_type + '_' + forecast_type_value;
}

string cache::UniqueNameFromOptions(search_options& options)
{
	assert(options.configuration->DatabaseType() == kNoDatabase || options.prod.Id() != kHPMissingInt);
	string producer_id = boost::lexical_cast<string>(options.prod.Id());
	string forecast_time = options.time.OriginDateTime().String("%Y-%m-%d_%H:%M:%S");
	string valid_time = options.time.ValidDateTime().String("%Y-%m-%d_%H:%M:%S");
	string param = options.param.Name();
	string level_value = boost::lexical_cast<string>((options.level).Value());
	string level = HPLevelTypeToString.at(options.level.Type());
	string forecast_type = boost::lexical_cast<string>(options.ftype.Type());
	string forecast_type_value = boost::lexical_cast<string>(options.ftype.Value());
	return producer_id + "_" + forecast_time + '_' + valid_time + '_' + param + '_' + level + '_' + level_value + '_' +
	       forecast_type + '_' + forecast_type_value;
}

void cache::Insert(info& anInfo, bool pin) { SplitToPool(anInfo, pin); }
void cache::SplitToPool(info& anInfo, bool pin)
{
	auto localInfo = anInfo;

	// Cached data is never replaced by another data that has
	// the same uniqueName

	string uniqueName = UniqueName(localInfo);

	{
		Lock lock(itsCheckMutex);

		if (cache_pool::Instance()->Find(uniqueName))
		{
			// New item, but only one thread should insert it (prevent race condition)

			itsLogger.Trace("Data with key " + uniqueName + " already exists at cache");

			// Update timestamp of this cache item
			cache_pool::Instance()->UpdateTime(uniqueName);
			return;
		}
	}

#ifdef HAVE_CUDA
	if (localInfo.Grid()->IsPackedData())
	{
		itsLogger.Trace("Removing packed data from cached info");
		localInfo.Grid()->PackedData().Clear();
	}
#endif

	assert(!localInfo.Grid()->IsPackedData());

	vector<param> params;
	vector<level> levels;
	vector<forecast_time> times;
	vector<forecast_type> ftype;

	params.push_back(localInfo.Param());
	levels.push_back(localInfo.Level());
	times.push_back(localInfo.Time());
	ftype.push_back(localInfo.ForecastType());

	auto newInfo = make_shared<info>(localInfo);

	newInfo->Params(params);
	newInfo->Levels(levels);
	newInfo->Times(times);
	newInfo->ForecastTypes(ftype);
	newInfo->Create(localInfo.Grid());

	assert(uniqueName == UniqueName(*newInfo));

	{
		Lock lock(itsCheckMutex);
		cache_pool::Instance()->Insert(uniqueName, newInfo, pin);
	}
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

void cache::Clean() { cache_pool::Instance()->Clean(); }
size_t cache::Size() const { return cache_pool::Instance()->Size(); }
cache_pool* cache_pool::itsInstance = NULL;

cache_pool::cache_pool() : itsCacheLimit(-1) { itsLogger = logger("cache_pool"); }

cache_pool* cache_pool::Instance()
{
	if (!itsInstance)
	{
		itsInstance = new cache_pool();
	}

	return itsInstance;
}

void cache_pool::CacheLimit(int theCacheLimit) { itsCacheLimit = theCacheLimit; }
bool cache_pool::Find(const string& uniqueName) { return itsCache.count(uniqueName) > 0; }
void cache_pool::Insert(const string& uniqueName, shared_ptr<himan::info> anInfo, bool pin)
{
	Lock lock(itsInsertMutex);

	cache_item item;
	item.info = anInfo;
	item.access_time = time(nullptr);
	item.pinned = pin;

	itsCache.insert(pair<string, cache_item>(uniqueName, item));
	itsLogger.Trace("Data added to cache with name: " + uniqueName + ", pinned: " + boost::lexical_cast<string>(pin));

	if (itsCacheLimit > -1 && itsCache.size() > static_cast<size_t>(itsCacheLimit))
	{
		Clean();
	}
}

void cache_pool::UpdateTime(const std::string& uniqueName) { itsCache[uniqueName].access_time = time(nullptr); }
void cache_pool::Clean()
{
	Lock lock(itsDeleteMutex);

	assert(itsCacheLimit > 0);
	if (itsCache.size() <= static_cast<size_t>(itsCacheLimit))
	{
		return;
	}

	string oldestName;
	time_t oldestTime = INT_MAX;

	for (const auto& kv : itsCache)
	{
		if (kv.second.access_time < oldestTime && !kv.second.pinned)
		{
			oldestName = kv.first;
			oldestTime = kv.second.access_time;
		}
	}

	assert(!oldestName.empty());

	itsCache.erase(oldestName);
	itsLogger.Trace("Data cleared from cache: " + oldestName +
	                " with time: " + boost::lexical_cast<string>(oldestTime));
	itsLogger.Trace("Cache size: " + boost::lexical_cast<string>(itsCache.size()));
}

shared_ptr<himan::info> cache_pool::GetInfo(const string& uniqueName)
{
	Lock lock(itsGetMutex);

	return make_shared<info>(*itsCache[uniqueName].info);
}

size_t cache_pool::Size() const { return itsCache.size(); }
