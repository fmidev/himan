#pragma once

#include "grid.h"
#include "lambert_conformal_grid.h"
#include "lambert_equal_area_grid.h"
#include "latitude_longitude_grid.h"
#include "stereographic_grid.h"
#include "transverse_mercator_grid.h"
#include <mutex>

namespace himan
{
class grid_cache
{
   public:
	static grid_cache* Instance();
	void Insert(const std::string& name, std::unique_ptr<grid> grid);
	template <typename T, typename... Args>
	std::unique_ptr<grid> Get(Args... args) const;

   private:
	std::unique_ptr<grid> Get(const std::string& name) const;

	static std::unique_ptr<grid_cache> itsInstance;
	std::map<std::string, std::unique_ptr<grid>> itsGridCache;
	mutable std::mutex itsAccessMutex;
};

inline grid_cache* grid_cache::Instance()
{
	if (!itsInstance)
	{
		itsInstance = std::make_unique<grid_cache>();
	}

	return itsInstance.get();
}

inline void grid_cache::Insert(const std::string& name, std::unique_ptr<grid> grid)
{
	std::lock_guard<std::mutex> lock(itsAccessMutex);
	if (itsGridCache.find(name) == itsGridCache.end())
	{
		itsGridCache.insert(std::make_pair(name, std::move(grid)));
	}
}

inline std::unique_ptr<grid> grid_cache::Get(const std::string& name) const
{
	std::lock_guard<std::mutex> lock(itsAccessMutex);
	const auto it = itsGridCache.find(name);

	if (it == itsGridCache.end())
	{
		return nullptr;
	}

	return std::move(it->second->Clone());
}

std::string ToString(const himan::earth_shape<double>& es)
{
	return es.Proj4String();
}
std::string ToString(const himan::point& p)
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
