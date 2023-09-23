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

namespace
{
std::string FormatSize(size_t size)
{
	const static size_t Ki = 1024;
	const static size_t Mi = 1024 * 1024;
	const static size_t Gi = 1024 * 1024 * 1024;

	if (size < Mi)
	{
		return fmt::format("{:d}Ki", size / Ki);
	}
	else if (size >= Mi && size < Gi)
	{
		return fmt::format("{:d}Mi", size / Mi);
	}
	else if (size >= Gi)
	{
		return fmt::format("{:d}Gi", size / Gi);
	}
	return fmt::format("{}", size);
}
float Ratio(size_t a, size_t b)
{
	return 100 * static_cast<float>(a) / static_cast<float>(b);
}
}  // namespace

typedef lock_guard<mutex> Lock;

cache::cache()
{
	itsLogger = logger("cache");
}

himan::HPWriteStatus cache::Insert(shared_ptr<info<double>> anInfo, bool pin)
{
	return Insert<double>(anInfo, pin);
}

template <typename T>
himan::HPWriteStatus cache::Insert(shared_ptr<info<T>> anInfo, bool pin)
{
	auto localInfo = make_shared<info<T>>(*anInfo);

	// Cached data is never replaced by another data that has
	// the same uniqueName

	const string uniqueName = util::UniqueName<T>(*localInfo);

	if (cache_pool::Instance()->Exists(uniqueName))
	{
		itsLogger.Trace(fmt::format("Cache item {} already exists", uniqueName));

		// Update timestamp of this cache item
		cache_pool::Instance()->UpdateTime(uniqueName);
		return himan::HPWriteStatus::kFinished;
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
	return cache_pool::Instance()->Insert<T>(uniqueName, localInfo, pin);
}

template himan::HPWriteStatus cache::Insert<double>(shared_ptr<info<double>>, bool);
template himan::HPWriteStatus cache::Insert<float>(shared_ptr<info<float>>, bool);
template himan::HPWriteStatus cache::Insert<short>(shared_ptr<info<short>>, bool);
template himan::HPWriteStatus cache::Insert<unsigned char>(shared_ptr<info<unsigned char>>, bool);

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

size_t cache::Clean(CleanType cleanType)
{
	return cache_pool::Instance()->Clean(cleanType);
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

void cache::Remove(const std::string& uniqueName)
{
	cache_pool::Instance()->Remove(uniqueName);
}

cache_pool* cache_pool::itsInstance = NULL;

cache_pool::cache_pool() : itsCacheLimit(0)
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

void cache_pool::CacheLimit(size_t theCacheLimit)
{
	itsCacheLimit = theCacheLimit;
}
bool cache_pool::Exists(const string& uniqueName)
{
	Lock lock(itsAccessMutex);
	return itsCache.count(uniqueName) > 0;
}

template <typename T>
himan::HPWriteStatus cache_pool::Insert(const string& uniqueName, shared_ptr<himan::info<T>> anInfo, bool pin)
{
	cache_item item;
	item.info = anInfo;
	item.access_time = time(nullptr);
	item.pinned = pin;
	item.size_bytes = anInfo->Data().Size() * sizeof(T);

	if (pin && itsCacheLimit > 0 && (Size() + item.size_bytes) > itsCacheLimit)
	{
		// cache is full and a pinned info needs to be written
		size_t cleaned = Clean(CleanType::kExcess);
#ifdef SERIALIZATION
		// cache clean failed, activate spill mechanism
		if (pin && cleaned == 0)
		{
			return HPWriteStatus::kFailed;
		}
#endif
	}

	{
		Lock lock(itsAccessMutex);

		itsCache.insert(pair<string, cache_item>(uniqueName, item));

		itsLogger.Trace(fmt::format("New cache item: {} pinned: {} size: {}", uniqueName, pin, item.size_bytes));
	}

	if (itsCacheLimit > 0)
	{
		Clean();
	}

	return himan::HPWriteStatus::kFinished;
}

template himan::HPWriteStatus cache_pool::Insert<double>(const string&, shared_ptr<himan::info<double>>, bool);
template himan::HPWriteStatus cache_pool::Insert<float>(const string&, shared_ptr<himan::info<float>>, bool);
template himan::HPWriteStatus cache_pool::Insert<short>(const string&, shared_ptr<himan::info<short>>, bool);
template himan::HPWriteStatus cache_pool::Insert<unsigned char>(const string&, shared_ptr<himan::info<unsigned char>>,
                                                                bool);

template <typename T>
void cache_pool::Replace(const string& uniqueName, shared_ptr<himan::info<T>> anInfo, bool pin)
{
	cache_item item;
	item.info = anInfo;
	item.access_time = time(nullptr);
	item.pinned = pin;
	item.size_bytes = anInfo->Data().Size() * sizeof(T);

	// possible race condition ?

	const auto it = itsCache.find(uniqueName);

	if (it != itsCache.end())
	{
		{
			Lock lock(itsAccessMutex);
			itsCache[uniqueName] = item;
		}
		itsLogger.Trace(fmt::format("Replaced cache item {}", uniqueName));
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

void cache_pool::Remove(const string& uniqueName)
{
	{
		Lock lock(itsAccessMutex);

		const auto it = itsCache.find(uniqueName);
		itsCache.erase(it);
	}
	itsLogger.Trace(fmt::format("Cache item {} removed", uniqueName));
}

void cache_pool::UpdateTime(const std::string& uniqueName)
{
	Lock lock(itsAccessMutex);
	itsCache[uniqueName].access_time = time(nullptr);
}

size_t cache_pool::Clean(CleanType cleanType)
{
	ASSERT(itsCacheLimit > 0);
	size_t cleaned = 0;

	if (cleanType == CleanType::kAll)
	{
		cleaned = Size();
		Lock lock(itsAccessMutex);
		itsCache.clear();
		return cleaned;
	}

	if (Size() <= itsCacheLimit)
	{
		return cleaned;
	}

	while (Size() > itsCacheLimit)
	{
		string oldestName;
		time_t oldestTime = INT_MAX;
		size_t oldestSize = 0;

		{
			Lock lock(itsAccessMutex);

			for (const auto& kv : itsCache)
			{
				if (kv.second.access_time < oldestTime && !kv.second.pinned)
				{
					oldestName = kv.first;
					oldestTime = kv.second.access_time;
					oldestSize = kv.second.size_bytes;
				}
			}

			if (oldestTime != INT_MAX)
			{
				itsCache.erase(oldestName);
			}
		}

		cleaned += oldestSize;

		const size_t sz = Size();

		if (oldestTime == INT_MAX)
		{
			itsLogger.Warning(fmt::format("Cannot clear cache, all existing data is pinned. Cache size {}/{} ({:.1f}%)",
			                              FormatSize(sz), FormatSize(itsCacheLimit), Ratio(sz, itsCacheLimit)));
			break;
		}
		itsLogger.Trace(fmt::format("Cache item {} with time {} removed", oldestName, oldestTime));
		itsLogger.Trace(fmt::format("Cache size: {}/{} ({:.1f}%)", FormatSize(sz), FormatSize(itsCacheLimit),
		                            Ratio(sz, itsCacheLimit)));
	}
	return cleaned;
}

namespace
{
template <typename T>
struct cache_visitor
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
		return make_shared<info<T>>(*std::get<std::shared_ptr<info<T>>>(it->second.info));
	}
	catch (const std::bad_variant_access& e)
	{
		if (strict)
		{
			return nullptr;
		}

		// convert to wanted data type
		return std::visit(cache_visitor<T>(), it->second.info);
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
	size_t bytes = 0;
	for (const auto& m : itsCache)
	{
		bytes += m.second.size_bytes;
	}
	return bytes;
}
