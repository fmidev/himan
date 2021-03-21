/**
 * @file cache.h
 *
 */

#ifndef CACHE_H
#define CACHE_H

#include "auxiliary_plugin.h"
#include "info.h"
#include "search_options.h"
#include <boost/variant.hpp>
#include <mutex>

namespace himan
{
namespace plugin
{
struct cache_item
{
	boost::variant<std::shared_ptr<himan::info<double>>, std::shared_ptr<himan::info<float>>,
	               std::shared_ptr<himan::info<short>>, std::shared_ptr<himan::info<unsigned char>>>

	    info;
	time_t access_time;
	bool pinned;

	cache_item() : access_time(0), pinned(false)
	{
	}
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
	void Insert(std::shared_ptr<info<T>> anInfo, bool pin = false);

	void Insert(std::shared_ptr<info<double>> anInfo, bool pin = false);

	template <typename T>
	std::vector<std::shared_ptr<info<T>>> GetInfo(search_options& options, bool strict = false);

	std::vector<std::shared_ptr<info<double>>> GetInfo(search_options& options, bool strict = false);

	template <typename T>
	std::vector<std::shared_ptr<info<T>>> GetInfo(const std::string& uniqueName, bool strict = false);

	void Clean();

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
	void Insert(const std::string& uniqueName, std::shared_ptr<info<T>> info, bool pin);

	/**
	 * @brief Get info from cache
	 *
	 * @param uniqueName unique label that identifies a cache element
	 * @param strict define whether cache is allowed to do data type conversion (--> strict=false)
	 */

	template <typename T>
	std::shared_ptr<info<T>> GetInfo(const std::string& uniqueName, bool strict);

	void Clean();

	virtual std::string ClassName() const override
	{
		return "himan::plugin::cache_pool";
	};
	virtual HPPluginClass PluginClass() const override
	{
		return kAuxiliary;
	};
	void UpdateTime(const std::string& uniqueName);
	void CacheLimit(int theCacheLimit);

	/**
	 * @brief Return current cache size (number of elements)
	 */

	size_t Size() const;

	/**
	 * @brief Replaces an element in the cache.
	 *
	 * If element is not found, insert is made.
	 */

	template <typename T>
	void Replace(const std::string& uniqueName, std::shared_ptr<info<T>> info, bool pin);

   private:
	cache_pool();

	std::map<std::string, cache_item> itsCache;
	static cache_pool* itsInstance;
	std::mutex itsAccessMutex;

	// Cache limit specifies how many grids are held in the cache.
	// When limit is reached, oldest grids are automatically pruned.
	// Value of -1 means no limit, 0 is not allowed (since there is a
	// separate configuration option to prevent himan from using cache)

	int itsCacheLimit;
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
