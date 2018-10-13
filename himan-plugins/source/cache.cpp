/**
 * @file cache.cpp
 *
 */

#include "cache.h"
#include "info.h"
#include "logger.h"
#include "plugin_factory.h"
#include <time.h>

using namespace std;
using namespace himan::plugin;

typedef lock_guard<mutex> Lock;

cache::cache()
{
	itsLogger = logger("cache");
}

template <typename T>
string cache::UniqueName(const info<T>& info)
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

template string cache::UniqueName(const info<double>&);

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

	const string uniqueName = UniqueName(*localInfo);

	if (cache_pool::Instance()->Exists(uniqueName))
	{
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

template void cache::Insert<double>(shared_ptr<info<double>> anInfo, bool pin);
template void cache::Insert<float>(shared_ptr<info<float>> anInfo, bool pin);

vector<shared_ptr<himan::info<double>>> cache::GetInfo(search_options& options)
{
	return GetInfo<double>(options);
}

template <typename T>
vector<shared_ptr<himan::info<T>>> cache::GetInfo(search_options& options)
{
	string uniqueName = UniqueNameFromOptions(options);

	vector<shared_ptr<himan::info<T>>> infos;

	auto foundInfo = cache_pool::Instance()->GetInfo<T>(uniqueName);
	if (foundInfo)
	{
		infos.push_back(foundInfo);
	}

	itsLogger.Trace("Data " + string(foundInfo ? "found" : "not found") + " for " + uniqueName);

	return infos;
}

template vector<shared_ptr<himan::info<double>>> cache::GetInfo<double>(search_options&);
template vector<shared_ptr<himan::info<float>>> cache::GetInfo<float>(search_options&);

void cache::Clean()
{
	cache_pool::Instance()->Clean();
}
size_t cache::Size() const
{
	return cache_pool::Instance()->Size();
}

void cache::Replace(info_t anInfo, bool pin)
{
	auto localInfo = make_shared<info<double>>(*anInfo);

	if (localInfo->DimensionSize() > 1)
	{
		auto newInfo = make_shared<info<double>>(localInfo->ForecastType(), localInfo->Time(), localInfo->Level(),
		                                         localInfo->Param());
		newInfo->Base(localInfo->Base());
		localInfo = newInfo;
	}

	cache_pool::Instance()->Replace(UniqueName(*localInfo), localInfo, pin);
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
template <typename T, typename U>
shared_ptr<himan::info<U>> Convert(const shared_ptr<himan::info<T>>& fnd)
{
	// Since data type needs to be changed, we have to create new info
	auto ret = make_shared<himan::info<U>>(fnd->ForecastType(), fnd->Time(), fnd->Level(), fnd->Param());
	ret->Producer(fnd->Producer());

	auto b = make_shared<himan::base<U>>();
	b->grid = shared_ptr<himan::grid>(fnd->Grid()->Clone());
	b->data.Resize(fnd->Data().SizeX(), fnd->Data().SizeY());

	const auto& src = VEC(fnd);
	auto& dst = b->data.Values();

	copy(src.begin(), src.end(), dst.begin());

	replace_if(dst.begin(), dst.end(), [](const U& val) { return ::isnan(val); }, himan::MissingValue<U>());
	ret->Create(b);

	return ret;
}

template <typename T>
struct cache_visitor : public boost::static_visitor<std::shared_ptr<himan::info<T>>>
{
   public:
	shared_ptr<himan::info<T>> operator()(shared_ptr<himan::info<double>>& fnd) const
	{
		return Convert<double, T>(fnd);
	}
	shared_ptr<himan::info<T>> operator()(shared_ptr<himan::info<float>>& fnd) const
	{
		return Convert<float, T>(fnd);
	}
};
}  // namespace

template <typename T>
shared_ptr<himan::info<T>> cache_pool::GetInfo(const string& uniqueName)
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
		return boost::get<std::shared_ptr<info<T>>>(it->second.info);
	}
	catch (boost::bad_get& e)
	{
		itsLogger.Info("Found data from cache with different datatype, conversion imminent");

		return boost::apply_visitor(cache_visitor<T>(), it->second.info);
	}
	catch (exception& e)
	{
		itsLogger.Error(e.what());
	}

	return nullptr;
}

template shared_ptr<himan::info<double>> cache_pool::GetInfo<double>(const string&);
template shared_ptr<himan::info<float>> cache_pool::GetInfo<float>(const string&);

size_t cache_pool::Size() const
{
	return itsCache.size();
}
