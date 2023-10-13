/**
 * @file cache.h
 *
 */

#ifndef CACHE_H
#define CACHE_H

#include "auxiliary_plugin.h"
#include "info.h"
#include "search_options.h"
#include <mutex>
#include <variant>

namespace himan
{
namespace plugin
{
struct cache_item
{
	std::variant<std::shared_ptr<himan::info<double>>, std::shared_ptr<himan::info<float>>,
	             std::shared_ptr<himan::info<short>>, std::shared_ptr<himan::info<unsigned char>>>

	    info;
	// time when this data was last accessed
	time_t access_time;
	// if pinned data is not evicted ever
	bool pinned;
	// size of data in bytes
	size_t size_bytes;

	cache_item() : access_time(0), pinned(false), size_bytes(0)
	{
	}
};

enum class CleanType
{
	kExcess,
	kAll
};

class cache : public auxiliary_plugin
{
   public:
	cache();
	~cache() = default;
	cache(const cache& other) = delete;
	cache& operator=(const cache& other) = delete;

	/**
	 * @brief Insert data to cache
	 *
	 * @param anInfo Info class instance containing the data
	 * @param activeOnly Specify if we want to copy only the active part of the info class
	 * to cache
	 */

	template <typename T>
	HPWriteStatus Insert(std::shared_ptr<info<T>> anInfo, bool pin = false);

	HPWriteStatus Insert(std::shared_ptr<info<double>> anInfo, bool pin = false);

	template <typename T>
	std::vector<std::shared_ptr<info<T>>> GetInfo(search_options& options, bool strict = false);

	std::vector<std::shared_ptr<info<double>>> GetInfo(search_options& options, bool strict = false);

	template <typename T>
	std::vector<std::shared_ptr<info<T>>> GetInfo(const std::string& uniqueName, bool strict = false);

	size_t Clean(CleanType type = CleanType::kExcess);

	virtual std::string ClassName() const override
	{
		return "himan::plugin::cache";
	};
	virtual HPPluginClass PluginClass() const override
	{
		return kAuxiliary;
	};

	size_t Size() const;

	template <typename T>
	void Replace(std::shared_ptr<info<T>> anInfo, bool pin = false);

	void Replace(std::shared_ptr<info<double>> anInfo, bool pin = false);

	void Remove(const std::string& uniqueName);

   private:
};

class cache_pool : public auxiliary_plugin
{
   public:
	~cache_pool()
	{
		if (itsInstance)
		{
			delete itsInstance;
		}
	}
	cache_pool(const cache_pool& other) = delete;
	cache_pool& operator=(const cache_pool& other) = delete;

	static cache_pool* Instance();
	bool Exists(const std::string& uniqueName);

	template <typename T>
	HPWriteStatus Insert(const std::string& uniqueName, std::shared_ptr<info<T>> info, bool pin);

	/**
	 * @brief Get info from cache
	 *
	 * @param uniqueName unique label that identifies a cache element
	 * @param strict define whether cache is allowed to do data type conversion (--> strict=false)
	 */

	template <typename T>
	std::shared_ptr<info<T>> GetInfo(const std::string& uniqueName, bool strict);

	size_t Clean(CleanType type = CleanType::kExcess);

	virtual std::string ClassName() const override
	{
		return "himan::plugin::cache_pool";
	};
	virtual HPPluginClass PluginClass() const override
	{
		return kAuxiliary;
	};
	void UpdateTime(const std::string& uniqueName);
	void CacheLimit(size_t theCacheLimit);

	/**
	 * @brief Return current cache size in bytes)
	 */

	size_t Size() const;

	/**
	 * @brief Replaces an element in the cache.
	 *
	 * If element is not found, insert is made.
	 */

	template <typename T>
	void Replace(const std::string& uniqueName, std::shared_ptr<info<T>> info, bool pin);

	void Remove(const std::string& uniqueName);

   private:
	cache_pool();

	std::map<std::string, cache_item> itsCache;
	static cache_pool* itsInstance;
	std::mutex itsAccessMutex;

	// Cache limit specifies how many grids are held in the cache.
	// When limit is reached, oldest grids are automatically pruned.
	// Value of 0 means no limit.

	size_t itsCacheLimit;
};

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory
extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<cache>();
}
#define HIMAN_AUXILIARY_INCLUDE
#endif /* HIMAN_AUXILIARY_INCLUDE */

}  // namespace plugin
}  // namespace himan

#endif /* CACHE_H */
