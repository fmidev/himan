/**
 * @file cache.h
 *
 */

#ifndef CACHE_H
#define CACHE_H

#include "auxiliary_plugin.h"
#include "search_options.h"

#include <mutex>

namespace himan
{
namespace plugin
{
struct cache_item
{
	std::shared_ptr<himan::info> info;
	time_t access_time;
	bool pinned;

	cache_item() : access_time(0), pinned(false) {}
};

class cache : public auxiliary_plugin
{
   public:
	cache();
	~cache() {}
	cache(const cache& other) = delete;
	cache& operator=(const cache& other) = delete;

	/**
	 * @brief Insert data to cache
	 *
	 * @param anInfo Info class instance containing the data
	 * @param activeOnly Specify if we want to copy only the active part of the info class
	 * to cache
	 */

	void Insert(info& anInfo, bool pin = false);
	std::vector<std::shared_ptr<himan::info>> GetInfo(search_options& options);
	void Clean();

	virtual std::string ClassName() const { return "himan::plugin::cache"; };
	virtual HPPluginClass PluginClass() const { return kAuxiliary; };
	virtual HPVersionNumber Version() const { return HPVersionNumber(1, 2); }
	size_t Size() const;

   private:
	void SplitToPool(info& anInfo, bool pin);
	std::string UniqueName(const info& anInfo);
	std::string UniqueNameFromOptions(search_options& options);
};

class cache_pool : public auxiliary_plugin
{
   public:
	~cache_pool() { delete itsInstance; }
	cache_pool(const cache_pool& other) = delete;
	cache_pool& operator=(const cache_pool& other) = delete;

	static cache_pool* Instance();
	bool Find(const std::string& uniqueName);
	void Insert(const std::string& uniqueName, std::shared_ptr<himan::info> info, bool pin);
	std::shared_ptr<himan::info> GetInfo(const std::string& uniqueName);
	void Clean();

	virtual std::string ClassName() const { return "himan::plugin::cache_pool"; };
	virtual HPPluginClass PluginClass() const { return kAuxiliary; };
	virtual HPVersionNumber Version() const { return HPVersionNumber(1, 1); }
	void UpdateTime(const std::string& uniqueName);
	void CacheLimit(int theCacheLimit);

	/**
	 * @brief Return current cache size (number of elements)
	 */

	size_t Size() const;

   private:
	cache_pool();

	std::map<std::string, cache_item> itsCache;
	static cache_pool* itsInstance;
	std::mutex itsInsertMutex;
	std::mutex itsGetMutex;
	std::mutex itsDeleteMutex;

	// Cache limit specifies how many grids are held in the cache.
	// When limit is reached, oldest grids are automatically pruned.
	// Value of -1 means no limit, 0 is not allowed (since there is a
	// separate configuration option to prevent himan from using cache)

	int itsCacheLimit;
};

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory
extern "C" std::shared_ptr<himan_plugin> create() { return std::shared_ptr<cache>(new cache()); }
#define HIMAN_AUXILIARY_INCLUDE
#endif /* HIMAN_AUXILIARY_INCLUDE */

}  // namespace plugin
}  // namespace himan

#endif /* CACHE_H */
