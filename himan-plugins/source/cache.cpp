/**
 * @file cache.cpp
 *
 */

#include "cache.h"
#include "info.h"
#include "logger.h"
#include "plugin_factory.h"
#include "util.h"
#include <time.h>

using namespace std;
using namespace himan::plugin;

typedef lock_guard<mutex> Lock;

cache::cache()
{
	itsLogger = logger("cache");
}

void cache::Insert(shared_ptr<info<double>> anInfo, bool pin)
{
	return Insert<double>(anInfo, pin);
}

template <typename T>
void cache::Insert(shared_ptr<info<T>> anInfo, bool pin)
{
	auto localInfo = make_shared<info<T>>(*anInfo);

	// Cached data is never replaced by another data that has
	// the same uniqueName

	const string uniqueName = util::UniqueName<T>(*localInfo);

	if (cache_pool::Instance()->Exists(uniqueName))
	{
		// TODO: should we replace existing item?
		itsLogger.Trace("Data with key " + uniqueName + " already exists at cache");

		// Update timestamp of this cache item
		cache_pool::Instance()->UpdateTime(uniqueName);
		return;
	}

#ifdef HAVE_CUDA
	if (localInfo->PackedData()->HasData())
	{
		itsLogger.Trace("Removing packed data from cached info");
		localInfo->PackedData()->Clear();
	}
#endif

	ASSERT(localInfo->PackedData()->HasData() == false);

	// localInfo might contain multiple grids. When adding data to cache, we need
	// to make sure that single info contains only single grid.

	if (localInfo->DimensionSize() > 1)
	{
		auto newInfo =
		    make_shared<info<T>>(localInfo->ForecastType(), localInfo->Time(), localInfo->Level(), localInfo->Param());
		newInfo->Producer(localInfo->Producer());
		newInfo->Base(localInfo->Base());
		localInfo = newInfo;
	}

	ASSERT(localInfo->DimensionSize() == 1);
	cache_pool::Instance()->Insert<T>(uniqueName, localInfo, pin);
}

template void cache::Insert<double>(shared_ptr<info<double>>, bool);
template void cache::Insert<float>(shared_ptr<info<float>>, bool);
template void cache::Insert<short>(shared_ptr<info<short>>, bool);
template void cache::Insert<unsigned char>(shared_ptr<info<unsigned char>>, bool);

vector<shared_ptr<himan::info<double>>> cache::GetInfo(search_options& options, bool strict)
{
	return GetInfo<double>(options, strict);
}

template <typename T>
vector<shared_ptr<himan::info<T>>> cache::GetInfo(search_options& options, bool strict)
{
	const string uniqueName = util::UniqueName(options);

	return GetInfo<T>(uniqueName, strict);
}

template vector<shared_ptr<himan::info<double>>> cache::GetInfo<double>(search_options&, bool);
template vector<shared_ptr<himan::info<float>>> cache::GetInfo<float>(search_options&, bool);
template vector<shared_ptr<himan::info<short>>> cache::GetInfo<short>(search_options&, bool);
template vector<shared_ptr<himan::info<unsigned char>>> cache::GetInfo<unsigned char>(search_options&, bool);

template <typename T>
vector<shared_ptr<himan::info<T>>> cache::GetInfo(const string& uniqueName, bool strict)
{
	vector<shared_ptr<himan::info<T>>> infos;

	auto foundInfo = cache_pool::Instance()->GetInfo<T>(uniqueName, strict);
	if (foundInfo)
	{
		infos.push_back(foundInfo);
	}

	itsLogger.Trace("Data " + string(foundInfo ? "found" : "not found") + " for " + uniqueName);

	return infos;
}

template vector<shared_ptr<himan::info<double>>> cache::GetInfo<double>(const string&, bool);
template vector<shared_ptr<himan::info<float>>> cache::GetInfo<float>(const string&, bool);
template vector<shared_ptr<himan::info<short>>> cache::GetInfo<short>(const string&, bool);
template vector<shared_ptr<himan::info<unsigned char>>> cache::GetInfo<unsigned char>(const string&, bool);

void cache::Clean()
{
	cache_pool::Instance()->Clean();
}
size_t cache::Size() const
{
	return cache_pool::Instance()->Size();
}

void cache::Replace(shared_ptr<info<double>> anInfo, bool pin)
{
	return Replace<double>(anInfo, pin);
}

template <typename T>
void cache::Replace(shared_ptr<info<T>> anInfo, bool pin)
{
	auto localInfo = make_shared<info<T>>(*anInfo);

	if (localInfo->DimensionSize() > 1)
	{
		auto newInfo =
		    make_shared<info<T>>(localInfo->ForecastType(), localInfo->Time(), localInfo->Level(), localInfo->Param());
		newInfo->Producer(localInfo->Producer());
		newInfo->Base(localInfo->Base());
		localInfo = newInfo;
	}

	cache_pool::Instance()->Replace<T>(util::UniqueName<T>(*localInfo), localInfo, pin);
}

template void cache::Replace<double>(shared_ptr<info<double>>, bool);
template void cache::Replace<float>(shared_ptr<info<float>>, bool);
template void cache::Replace<short>(shared_ptr<info<short>>, bool);
template void cache::Replace<unsigned char>(shared_ptr<info<unsigned char>>, bool);

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
bool cache_pool::Exists(const string& uniqueName)
{
	Lock lock(itsAccessMutex);
	return itsCache.count(uniqueName) > 0;
}

template <typename T>
void cache_pool::Insert(const string& uniqueName, shared_ptr<himan::info<T>> anInfo, bool pin)
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

template void cache_pool::Insert<double>(const string&, shared_ptr<himan::info<double>>, bool);
template void cache_pool::Insert<float>(const string&, shared_ptr<himan::info<float>>, bool);
template void cache_pool::Insert<short>(const string&, shared_ptr<himan::info<short>>, bool);
template void cache_pool::Insert<unsigned char>(const string&, shared_ptr<himan::info<unsigned char>>, bool);

template <typename T>
void cache_pool::Replace(const string& uniqueName, shared_ptr<himan::info<T>> anInfo, bool pin)
{
	cache_item item;
	item.info = anInfo;
	item.access_time = time(nullptr);
	item.pinned = pin;

	// possible race condition ?

	const auto it = itsCache.find(uniqueName);

	if (it != itsCache.end())
	{
		{
			Lock lock(itsAccessMutex);
			itsCache[uniqueName] = item;
		}
		itsLogger.Trace("Data with name " + uniqueName + " replaced");
	}
	else
	{
		Insert<T>(uniqueName, anInfo, pin);
	}
}

template void cache_pool::Replace<double>(const string&, shared_ptr<himan::info<double>>, bool);
template void cache_pool::Replace<float>(const string&, shared_ptr<himan::info<float>>, bool);
template void cache_pool::Replace<short>(const string&, shared_ptr<himan::info<short>>, bool);
template void cache_pool::Replace<unsigned char>(const string&, shared_ptr<himan::info<unsigned char>>, bool);

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

namespace
{
template <typename T>
struct cache_visitor : public boost::static_visitor<std::shared_ptr<himan::info<T>>>
{
   public:
	shared_ptr<himan::info<T>> operator()(shared_ptr<himan::info<double>>& fnd) const
	{
		return make_shared<himan::info<T>>(*fnd);
	}
	shared_ptr<himan::info<T>> operator()(shared_ptr<himan::info<float>>& fnd) const
	{
		return make_shared<himan::info<T>>(*fnd);
	}
	shared_ptr<himan::info<T>> operator()(shared_ptr<himan::info<short>>& fnd) const
	{
		return make_shared<himan::info<T>>(*fnd);
	}
	shared_ptr<himan::info<T>> operator()(shared_ptr<himan::info<unsigned char>>& fnd) const
	{
		return make_shared<himan::info<T>>(*fnd);
	}
};
}  // namespace

template <typename T>
shared_ptr<himan::info<T>> cache_pool::GetInfo(const string& uniqueName, bool strict)
{
	std::map<std::string, cache_item>::iterator it;

	{
		Lock lock(itsAccessMutex);
		it = itsCache.find(uniqueName);
	}

	if (it == itsCache.end())
	{
		return nullptr;
	}

	try
	{
		// if data is directly found from cache with correct data type,
		// return that
		return make_shared<info<T>>(*boost::get<std::shared_ptr<info<T>>>(it->second.info));
	}
	catch (boost::bad_get& e)
	{
		if (strict)
		{
			return nullptr;
		}

		// convert to wanted data type
		return boost::apply_visitor(cache_visitor<T>(), it->second.info);
	}
	catch (exception& e)
	{
		itsLogger.Error(e.what());
	}

	return nullptr;
}

template shared_ptr<himan::info<double>> cache_pool::GetInfo<double>(const string&, bool);
template shared_ptr<himan::info<float>> cache_pool::GetInfo<float>(const string&, bool);
template shared_ptr<himan::info<short>> cache_pool::GetInfo<short>(const string&, bool);
template shared_ptr<himan::info<unsigned char>> cache_pool::GetInfo<unsigned char>(const string&, bool);

size_t cache_pool::Size() const
{
	return itsCache.size();
}
