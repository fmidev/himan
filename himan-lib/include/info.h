/**
 * @file info.h
 *
 * @brief Define metadata structures requred to calculate and store data
 *
 */

#ifndef INFO_H
#define INFO_H

#include "forecast_time.h"
#include "forecast_type.h"
#include "grid.h"
#include "himan_common.h"
#include "lambert_conformal_grid.h"
#include "latitude_longitude_grid.h"
#include "level.h"
#include "matrix.h"
#include "packed_data.h"
#include "param.h"
#include "point_list.h"
#include "producer.h"
#include "raw_time.h"
#include "serialization.h"
#include "station.h"
#include "stereographic_grid.h"
#include <limits>  // for std::numeric_limits<size_t>::max();
#include <vector>

#define VEC(I) I->Data().Values()

namespace himan
{
namespace plugin
{
class compiled_plugin_base;
}

/**
 * @class iterator
 *
 * @brief Nested class inside info to provide iterator functions to info class
 *
 */

const size_t kIteratorResetValue = std::numeric_limits<size_t>::max();

template <class U>
class iterator
{
   public:
	iterator<U>() : itsIndex(kIteratorResetValue)
	{
	}
	explicit iterator<U>(const std::vector<U>& theElements) : itsElements(theElements)
	{
		Reset();
	}
	explicit iterator(const iterator& other) : itsElements(other.itsElements), itsIndex(other.itsIndex)
	{
	}
	iterator& operator=(const iterator& other)
	{
		itsElements = other.itsElements;
		itsIndex = other.itsIndex;
		return *this;
	}

	std::string ClassName() const
	{
		return "himan::iterator";
	}
	/**
	 * @brief Reset iterator
	 *
	 * Reset iterator by setting index value to max size_t (which equals to max unsigned int)
	 *
	 * @return void
	 *
	 */

	void Reset()
	{
		itsIndex = kIteratorResetValue;
	}
	/**
	 * @brief Set iterator to first element
	 *
	 * @return boolean if iterator has at least one element, else false
	 *
	 */

	bool First()
	{
		Reset();

		return Next();
	}

	/**
	 * @brief Set iterator to first element
	 *
	 * @return boolean if iterator has at least one element, else false
	 *
	 */

	bool Last()
	{
		Reset();

		return Previous();
	}

	/**
	 * @brief Retreat iterator by one
	 *
	 * @return boolean if iterator has more elements left, else false
	 *
	 */

	bool Previous()
	{
		if (itsElements.size() == 0)
		{
			return false;
		}

		if (itsIndex == kIteratorResetValue)
		{
			itsIndex =
			    itsElements.size() == 0 ? 0 : itsElements.size() - 1;  // Reset() has been called before this function
		}

		else if (itsIndex == 0)
		{
			// already at first element
			return false;
		}

		else
		{
			itsIndex--;
		}

		return true;
	}

	/**
	 * @brief Advance iterator by one
	 *
	 * @return boolean if iterator has more elements left, else false
	 *
	 */

	bool Next()
	{
		if (itsElements.size() == 0)
		{
			return false;
		}

		if (itsIndex == kIteratorResetValue)
		{
			itsIndex = 0;  // Reset() has been called before this function
		}

		else if (itsIndex >= (itsElements.size() - 1))
		{
			// already at last element
			return false;
		}

		else
		{
			itsIndex++;
		}

		return true;
	}

	/**
	 * @return Reference to current value or throw exception
	 */

	U& At()
	{
		if (itsIndex != kIteratorResetValue && itsIndex < itsElements.size())
		{
			return itsElements[itsIndex];
		}

		std::cerr << ClassName() + ": Invalid index value: " + std::to_string(itsIndex) << std::endl;
		himan::Abort();
	}

	const U& At() const
	{
		if (itsIndex != kIteratorResetValue && itsIndex < itsElements.size())
		{
			return itsElements[itsIndex];
		}

		std::cerr << ClassName() + ": Invalid index value: " + std::to_string(itsIndex) << std::endl;
		himan::Abort();
	}

	/**
	 * @return Reference to value requested or throw exception
	 */

	const U& At(size_t theIndex) const
	{
		if (theIndex < itsElements.size())
		{
			return itsElements[theIndex];
		}

		std::cerr << ClassName() + ": Invalid index value: " + std::to_string(itsIndex) << std::endl;
		himan::Abort();
	}

	/**
	 * @brief Set iterator to the position indicated by the function argument
	 *
	 * @return True if value exists, else false
	 *
	 */

	bool Set(const U& theElement)
	{
		for (size_t i = 0; i < itsElements.size(); i++)
		{
			if (itsElements[i] == theElement)
			{
				Set(i);
				return true;
			}
		}

		return false;
	}

	/**
	 * @brief Set iterator to the position indicated by the function argument. No limit-checking is made.
	 *
	 * @return void
	 *
	 * @todo Should return bool like Set(const T theElement) ?
	 */

	void Index(size_t theIndex)
	{
		itsIndex = theIndex;
	}
	// DEPRECATED
	void Set(size_t theIndex)
	{
		itsIndex = theIndex;
	}
	/**
	 * @brief Replace the value at current iterator position with a new value
	 *
	 */

	void Replace(const U& theNewValue)
	{
		itsElements.at(itsIndex) = theNewValue;
	}
	/**
	 * @return Current index value
	 */

	size_t Index() const
	{
		return itsIndex;
	}
	/**
	 * @return Iterator size
	 */

	size_t Size() const
	{
		return itsElements.size();
	}
	friend std::ostream& operator<<(std::ostream& file, const iterator<U>& ob)
	{
		return ob.Write(file);
	}
	/**
	 * @brief Add element to iterator
	 *
	 * NOTE: This function DOES NOT change the size of itsDimensions! That
	 * needs to be done separately!
	 *
	 * @param newElement Element to be added
	 * @param strict Define whether to allow duplicate values (false = allow)
	 * @return True if adding was successful
	 */

	bool Add(const U& newElement, bool strict = true)
	{
		if (strict)
		{
			size_t tempIndex = itsIndex;

			if (Set(newElement))
			{
				itsIndex = tempIndex;
				return false;
			}

			itsIndex = tempIndex;
		}

		itsElements.push_back(newElement);

		return true;
	}

	/**
	 * @brief Remove all elements (iterator size = 0)
	 */

	void Clear()
	{
		itsElements.clear();
	}

	std::vector<U> Values() const
	{
		return itsElements;
	}

	/**
	 * @brief Write object to stream
	 */

	std::ostream& Write(std::ostream& file) const
	{
		file << "<" << ClassName() << ">" << std::endl;
		file << "__itsIndex__ " << itsIndex << std::endl;
		file << "__itsSize__ " << itsElements.size() << std::endl;

		for (size_t i = 0; i < itsElements.size(); i++)
		{
			file << itsElements[i];
		}

		return file;
	}

   private:
	std::vector<U> itsElements;  //<! Vector to hold the elements
	size_t itsIndex;             //<! Current index of iterator

#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsElements), CEREAL_NVP(itsIndex));
	}
#endif
};

class forecast_time;
class param;
class level;
class logger;
class forecast_type;

typedef iterator<level> level_iter;
typedef iterator<param> param_iter;
typedef iterator<forecast_time> time_iter;
typedef iterator<producer> producer_iter;
typedef iterator<forecast_type> forecast_type_iter;

template <typename U>
struct type2type
{
	typedef U type;
};

template <typename T>
struct base
{
	std::shared_ptr<himan::grid> grid;
	matrix<T> data;
	std::shared_ptr<packed_data> pdata;

	base() : grid(), data(0, 0, 1, MissingValue<T>()), pdata(std::make_shared<packed_data>())
	{
	}
	base(std::shared_ptr<himan::grid> grid_, const matrix<T>& data_)
	    : grid(grid_), data(data_), pdata(std::make_shared<packed_data>())
	{
	}

#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		// packed data is not serialized as it contains raw pointers
		ar(CEREAL_NVP(grid), CEREAL_NVP(data));
	}
#endif
};

template <typename T>
class info
{
   public:
	friend class himan::plugin::compiled_plugin_base;
	template <typename>
	friend class info;  // for templated copy constructor

	info() = default;
	~info() = default;

	/**
	 * @brief Copy constructor for info class. Will preserve data backend.
	 * New info has the same data backend matrix as the original one.
	 * This means that multiple threads can access the same data with
	 * different infos ( --> descriptor positions ). Clone will have the
	 * same initial descriptor positions.
	 */

	info(const info& other);

	// 'coercion constructor' to create info from an info with a different data type
	template <typename V>
	info(const info<V>& other);

	info(const std::vector<forecast_type>& ftypes, const std::vector<forecast_time>& times,
	     const std::vector<level>& levels, const std::vector<param>& params);
	info(const forecast_type& ftype, const forecast_time& time, const level& level, const param& param);

	info& operator=(const info& other) = delete;

	std::string ClassName() const
	{
		return "himan::info";
	}
	std::ostream& Write(std::ostream& file) const;

	/**
	 * @brief Merge @param to 'this' info
	 * @param otherInfo
	 */

	void Merge(std::shared_ptr<info> otherInfo);

	/**
	 * @brief Merge all infos in @param to 'this' info
	 * @param otherInfos
	 */

	void Merge(std::vector<std::shared_ptr<info>>& otherInfos);

	/**
	 * @brief Initialize data backend with correct number of matrices
	 *
	 * Function will create a number of matrices to
	 * hold the data. The number of the matrices depends on the size
	 * of times, params and levels.
	 *
	 * Data is copied.
	 *
	 * Will *not* preserve iterator positions.
	 */

	void Create(std::shared_ptr<base<T>> baseGrid, bool createDataBackend = false);

	/**
	 * @brief Re-order infos in the dimension vector if dimension sizes are changed
	 *
	 * If existing dimensions are resized, the data in the dimension vector needs
	 * to be reordered or the calculated iterator indices are not correct and segfault
	 * is more than likely.
	 *
	 * Regridding will therefore *move* the grids from the old dimension vector to a
	 * new one. For now regridding is only supported for level and param dimensions.
	 *
	 * Will *not* preserve iterator positions.
	 */

	template <typename U>
	void Regrid(const std::vector<U>& newDim);

	void Producer(long theFmiProducerID)
	{
		itsProducer = producer(theFmiProducerID);
	}

	void Producer(const producer& theProducer)
	{
		itsProducer = theProducer;
	}

	const producer& Producer() const
	{
		return itsProducer;
	}

	//! Set all iterators to first position
	void First();

	//! Reset all iterators
	void Reset();

	/**
	 * @brief Advance dimension iterators in their natural order.
	 *
	 * Location iterator is not advanced
	 */

	bool Next();

	/**
	 * @brief Set location iterator to given index value. No limit-checking is made.
	 */

	void LocationIndex(size_t theLocationIndex)
	{
		itsLocationIndex = theLocationIndex;
	}

	void ResetLocation()
	{
		itsLocationIndex = kIteratorResetValue;
	}

	bool NextLocation();
	bool FirstLocation()
	{
		ResetLocation();
		return NextLocation();
	}
	size_t LocationIndex() const
	{
		return itsLocationIndex;
	}

	size_t SizeLocations() const
	{
		return itsDimensions[Index()]->data.Size();
	}

	/**
	 * @brief Return current latlon coordinates
	 *
	 * Does not currently support stereographic projection.
	 * In rotated latlon projection function return coordinates in rotated form.
	 *
	 * @return Latitude and longitude of current grid point
	 */

	point LatLon() const;

	/**
	 * @brief Return station information corresponding to current location. Only valid
	 * for irregular grids.
	 *
	 * @return station information
	 */

	station Station() const;

	/**
	 * @brief Replace current grid with the function argument
	 * @param d shared pointer to a grid instance
	 */

	void Base(std::shared_ptr<base<T>> b)
	{
		ASSERT(itsDimensions.size() > Index());
		itsDimensions[Index()] = b;
	}

	std::shared_ptr<base<T>> Base()
	{
		return itsDimensions[Index()];
	}

	/**
	 * @brief Shortcut to get the current data matrix
	 * @return Current data matrix
	 */

	matrix<T>& Data()
	{
		return itsDimensions[Index()]->data;
	}

	std::shared_ptr<grid> Grid() const
	{
		ASSERT(itsDimensions.size());
		return itsDimensions[Index()]->grid;
	}

	std::shared_ptr<packed_data> PackedData() const
	{
		return itsDimensions[Index()]->pdata;
	}

	/**
	 * @brief Return size of meta matrix. Is the same as times*params*levels.
	 *
	 */

	size_t DimensionSize() const
	{
		return itsDimensions.size();
	}

	/**
	 * @brief Set the data value pointed by the iterators with a new one
	 */

	void Value(T theValue)
	{
		Data().Set(itsLocationIndex, theValue);
	}

	/**
	 * @return Data value pointed by the iterators
	 */

	T Value() const
	{
		return itsDimensions[Index()]->data.At(itsLocationIndex);
	}

	/**
	 * @brief Clear info contents and iterators
	 *
	 * Does not free memory explicitly.
	 */

	void Clear();

	bool IsValidGrid() const
	{
		return (itsDimensions[Index()] != nullptr && Grid());
	}

	/**
	 * @brief Set the iterator positions to first valid grid found.
	 *
	 * If info sparsity=0 then it is the first iterator position.
	 * Will change iterator position only in the inner-most dimension
	 * (parameter).
	 */
	void FirstValidGrid();
	info Clone();

	// The following four functions are for convenience and backwards
	// compatibility only

	const param& Param() const
	{
		return itsParamIterator.At();
	}

	param& Param()
	{
		return itsParamIterator.At();
	}

	const level& Level() const
	{
		return itsLevelIterator.At();
	}

	level& Level()
	{
		return itsLevelIterator.At();
	}

	const forecast_time& Time() const
	{
		return itsTimeIterator.At();
	}

	forecast_time& Time()
	{
		return itsTimeIterator.At();
	}

	const forecast_type& ForecastType() const
	{
		return itsForecastTypeIterator.At();
	}

	forecast_type& ForecastType()
	{
		return itsForecastTypeIterator.At();
	}

	template <typename U>
	himan::iterator<U>& Iterator()
	{
		return ReturnIterator(type2type<U>());
	}

	template <typename U>
	const himan::iterator<U>& Iterator() const
	{
		return ReturnIterator(type2type<U>());
	}
	//! Set iterator position to first element
	template <typename U>
	bool First()
	{
		return Iterator<U>().First();
	}

	//! Set iterator position to last element
	template <typename U>
	bool Last()
	{
		return Iterator<U>().Last();
	}

	//! Advance iterator by one
	template <typename U>
	bool Next()
	{
		return Iterator<U>().Next();
	}

	//! Retreat iterator by one
	template <typename U>
	bool Previous()
	{
		return Iterator<U>().Previous();
	}

	//! Reset iterator position (not pointing to any element)
	template <typename U>
	void Reset()
	{
		Iterator<U>().Reset();
	}

	//! Replace iterator with a new one
	template <typename U>
	void Iterator(const himan::iterator<U>& theIter)
	{
		Iterator<U>() = theIter;
	}

	//! Set iterator values from a vector replacing old values
	template <typename U>
	void Set(const std::vector<U>& values)
	{
		auto& iter = Iterator<U>();

		if (!itsDimensions.empty() && iter.Size() && iter.Size() < values.size())
		{
			Regrid<U>(values);
		}

		iter = iterator<U>(values);
	}

	//! Replace a single value in iterator
	template <typename U>
	void Set(const U& value)
	{
		Iterator<U>().Replace(value);
	}

	//! Return iteraror value with given index without moving the position
	template <typename U>
	const U& Peek(size_t index) const
	{
		return Iterator<U>().At(index);
	}

	//! Find if a given value is in the iterator range
	template <typename U>
	bool Find(const U& value)
	{
		return Iterator<U>().Set(value);
	}

	//! Return current iterator index number
	template <typename U>
	size_t Index() const
	{
		return Iterator<U>().Index();
	}

	//! Set current iterator index number
	template <typename U>
	void Index(size_t theIndex)
	{
		Iterator<U>().Index(theIndex);
	}

	//! Return iterator size
	template <typename U>
	size_t Size() const
	{
		return Iterator<U>().Size();
	}

	//! Return iterator value from current position
	template <typename U>
	U Value() const
	{
		return Iterator<U>().At();
	}

   protected:
	std::vector<std::shared_ptr<base<T>>>& Dimensions()
	{
		return itsDimensions;
	}

   private:
	const himan::iterator<forecast_type>& ReturnIterator(type2type<forecast_type>) const
	{
		return itsForecastTypeIterator;
	}

	himan::iterator<forecast_type>& ReturnIterator(type2type<forecast_type>)
	{
		return itsForecastTypeIterator;
	}

	const himan::iterator<forecast_time>& ReturnIterator(type2type<forecast_time>) const
	{
		return itsTimeIterator;
	}

	himan::iterator<forecast_time>& ReturnIterator(type2type<forecast_time>)
	{
		return itsTimeIterator;
	}

	const himan::iterator<level>& ReturnIterator(type2type<level>) const
	{
		return itsLevelIterator;
	}

	himan::iterator<level>& ReturnIterator(type2type<level>)
	{
		return itsLevelIterator;
	}

	const himan::iterator<param>& ReturnIterator(type2type<param>) const
	{
		return itsParamIterator;
	}

	himan::iterator<param>& ReturnIterator(type2type<param>)
	{
		return itsParamIterator;
	}

	/**
	 * @brief Re-create indexing of elements in meta-matrix if the dimensions
	 * have changed.
	 *
	 * Because meta-matrix is a sparse matrix, if the dimension sizes are changed
	 * (for example with Merge()), ie adding parameters, levels, times to an info
	 * the ordering must also be changed accordingly.
	 *
	 * ReIndex() moves data around but does not copy (ie allocate new memory).
	 *
	 */

	void ReIndex(size_t oldForecastTypeSize, size_t oldTimeSize, size_t oldLevelSize, size_t oldParamSize);

	/**
	 * @brief Return running index number when given relative index for each
	 * three dimension
	 *
	 * @param timeIndex x-dimension index
	 * @param levelIndex y-dimension index
	 * @param paramIndex z-dimension index
	 * @return
	 */

	size_t Index(size_t forecastTypeIndex, size_t timeIndex, size_t levelIndex, size_t paramIndex) const;

	size_t Index() const
	{
		return Index(Index<forecast_type>(), Index<forecast_time>(), Index<level>(), Index<param>());
	}

	level_iter itsLevelIterator;
	time_iter itsTimeIterator;
	param_iter itsParamIterator;
	forecast_type_iter itsForecastTypeIterator;

	std::vector<std::shared_ptr<base<T>>> itsDimensions;

	logger itsLogger = logger("info");

	producer itsProducer;

	size_t itsLocationIndex = kIteratorResetValue;

#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsLevelIterator), CEREAL_NVP(itsTimeIterator), CEREAL_NVP(itsParamIterator),
		   CEREAL_NVP(itsForecastTypeIterator), CEREAL_NVP(itsDimensions), CEREAL_NVP(itsLogger),
		   CEREAL_NVP(itsProducer), CEREAL_NVP(itsLocationIndex));
	}
#endif
};

#include "info_impl.h"

}  // namespace himan

#endif /* INFO_H */
