#pragma once

#include "grid.h"
#include "lambert_conformal_grid.h"
#include "lambert_equal_area_grid.h"
#include "latitude_longitude_grid.h"
#include "stereographic_grid.h"
#include "transverse_mercator_grid.h"
#include <shared_mutex>

namespace himan
{
class grid_cache
{
   public:
	static grid_cache& Instance();
	template <typename T, typename... Args>
	std::unique_ptr<grid> Get(Args... args) const;

	grid_cache(const grid_cache&) = delete;
	grid_cache(grid_cache&&) = delete;
	grid_cache& operator=(const grid_cache&) = delete;
	grid_cache& operator=(grid_cache&&) = delete;

   protected:
	grid_cache() = default;
	~grid_cache() = default;

   private:
	std::unique_ptr<grid> Get(const std::string& name) const;
	void Insert(const std::string& name, std::unique_ptr<grid> grid);

	std::map<std::string, std::unique_ptr<grid>> itsGridCache;
	static std::shared_mutex itsAccessMutex;
};

inline grid_cache& grid_cache::Instance()
{
	static grid_cache itsInstance;

	return itsInstance;
}

inline void grid_cache::Insert(const std::string& name, std::unique_ptr<grid> grid)
{
	if (itsGridCache.find(name) == itsGridCache.end())
	{
		itsGridCache.insert(std::make_pair(name, std::move(grid)));
	}
}

inline std::unique_ptr<grid> grid_cache::Get(const std::string& name) const
{
	const auto it = itsGridCache.find(name);

	if (it == itsGridCache.end())
	{
		return nullptr;
	}

	return std::unique_ptr<grid>(it->second->Clone());
}

inline std::string ToString(const himan::earth_shape<double>& es)
{
	return es.Proj4String();
}

inline std::string ToString(const himan::point& p)
{
	return static_cast<std::string>(p);
}

template <typename T>
std::string ToString(T var)
{
	return fmt::format("{}", var);
}

template <typename T, typename... Args>
std::string ToString(T first, Args... args)
{
	return ToString(first) + ToString(args...);
}

template <typename T, typename... Args>
struct grid_create
{
	std::unique_ptr<grid> operator()(Args... args)
	{
		return nullptr;
	}
};

template <typename... Args>
struct grid_create<himan::latitude_longitude_grid, Args...>
{
	std::unique_ptr<grid> operator()(Args... args)
	{
		return std::make_unique<latitude_longitude_grid>(args...);
	}
};

template <typename... Args>
struct grid_create<himan::rotated_latitude_longitude_grid, Args...>
{
	std::unique_ptr<grid> operator()(Args... args)
	{
		return std::make_unique<rotated_latitude_longitude_grid>(args...);
	}
};

template <typename... Args>
struct grid_create<himan::lambert_conformal_grid, Args...>
{
	std::unique_ptr<grid> operator()(Args... args)
	{
		return std::make_unique<lambert_conformal_grid>(args...);
	}
};

template <typename... Args>
struct grid_create<himan::lambert_equal_area_grid, Args...>
{
	std::unique_ptr<grid> operator()(Args... args)
	{
		return std::make_unique<lambert_equal_area_grid>(args...);
	}
};

template <typename... Args>
struct grid_create<himan::transverse_mercator_grid, Args...>
{
	std::unique_ptr<grid> operator()(Args... args)
	{
		return std::make_unique<transverse_mercator_grid>(args...);
	}
};

template <typename... Args>
struct grid_create<himan::stereographic_grid, Args...>
{
	std::unique_ptr<grid> operator()(Args... args)
	{
		return std::make_unique<stereographic_grid>(args...);
	}
};

template <typename T, typename... Args>
std::unique_ptr<grid> grid_cache::Get(Args... args) const
{
	const std::string name = ToString(args...);

	{
		std::shared_lock<std::shared_mutex> lock(itsAccessMutex);

		// Take a read lock, and check if grid is already in cache

		std::unique_ptr<grid> ret = Get(name);

		if (ret != nullptr)
		{
			return ret;
		}
	}

	// Take a write lock, but before creating a new grid, check
	// cache once more
	std::lock_guard<std::shared_mutex> rlock(itsAccessMutex);

	std::unique_ptr<grid> ret = Get(name);

	if (ret != nullptr)
	{
		return ret;
	}

	grid_create<T, Args...> a;
	ret = a(args...);

	if (ret != nullptr)
	{
		const_cast<grid_cache*>(this)->Insert(name, ret->Clone());
	}
	return ret;
}

}  // namespace himan
