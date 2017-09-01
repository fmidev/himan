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
#include "info_simple.h"
#include "level.h"
#include "param.h"
#include "producer.h"
#include "raw_time.h"
#include "serialization.h"
#include "station.h"
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

template <class T>
class iterator
{
   public:
	iterator<T>() : itsIndex(kIteratorResetValue) {}
	explicit iterator<T>(const std::vector<T>& theElements) : itsElements(theElements) { Reset(); }
	explicit iterator(const iterator& other) : itsElements(other.itsElements), itsIndex(other.itsIndex) {}
	iterator& operator=(const iterator& other)
	{
		itsElements = other.itsElements;
		itsIndex = other.itsIndex;
		return *this;
	}

	std::string ClassName() const { return "himan::iterator"; }
	/**
	 * @brief Reset iterator
	 *
	 * Reset iterator by setting index value to max size_t (which equals to max unsigned int)
	 *
	 * @return void
	 *
	 */

	void Reset() { itsIndex = kIteratorResetValue; }
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

	const T& At() const
	{
		if (itsIndex != kIteratorResetValue && itsIndex < itsElements.size())
		{
			return itsElements[itsIndex];
		}

		throw std::runtime_error(ClassName() + ": Invalid index value: " + std::to_string(itsIndex));
	}

	/**
	 * @return Reference to value requested or throw exception
	 */

	const T& At(size_t theIndex) const
	{
		if (theIndex < itsElements.size())
		{
			return itsElements[theIndex];
		}

		throw std::runtime_error(ClassName() + ": Invalid index value: " + std::to_string(theIndex));
	}

	/**
	 * @brief Set iterator to the position indicated by the function argument
	 *
	 * @return True if value exists, else false
	 *
	 */

	bool Set(const T& theElement)
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

	void Set(size_t theIndex) { itsIndex = theIndex; }
	/**
	 * @brief Replace the value at current iterator position with a new value
	 *
	 */

	void Replace(const T& theNewValue) { itsElements[itsIndex] = theNewValue; }
	/**
	 * @return Current index value
	 */

	size_t Index() const { return itsIndex; }
	/**
	 * @return Iterator size
	 */

	size_t Size() const { return itsElements.size(); }
	friend std::ostream& operator<<(std::ostream& file, const iterator<T>& ob) { return ob.Write(file); }
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

	bool Add(const T& newElement, bool strict = true)
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

	void Clear() { itsElements.clear(); }
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
	std::vector<T> itsElements;  //<! Vector to hold the elements
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

class info
{
   public:
	friend class json_parser;
	friend class himan::plugin::compiled_plugin_base;

	info();
	~info();

	/**
	 * @brief Copy constructor for info class. Will preserve data backend.

	 * New info has the same data backend matrix as the original one.
	 * This means that multiple threads can access the same data with
	 * different infos ( --> descriptor positions ). Clone will have the
	 * same initial descriptor positions.
	 */

	info(const info& other);

	info& operator=(const info& other) = delete;

	std::string ClassName() const { return "himan::info"; }
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
	 * @brief Initialize parameter iterator with new parameters
	 * @param theParams A vector containing new parameter information for this info
	 */

	void Params(const std::vector<param>& theParams);

	/**
	 * @brief Replace current parameter iterator with a new one
	 * @param theParamIterator New parameter iterator
	 */

	void ParamIterator(const param_iter& theParamIterator);

	/**
	 * @brief Initialize level iterator with new levels
	 * @param theLevels A vector containing new level information for this info
	 */

	void Levels(const std::vector<level>& theLevels);

	/**
	 * @brief Replace current level iterator with a new one
	 * @param theLevelIterator New level iterator
	 */

	void LevelIterator(const level_iter& theLevelIterator);

	/**
	 * @brief Initialize time iterator with new times
	 * @param theTimes A vector containing new time information for this info
	 */

	void Times(const std::vector<forecast_time>& theTimes);

	/**
	 * @brief Replace current time iterator with a new one
	 * @param theTimeIterator New time iterator
	 */

	void TimeIterator(const time_iter& theTimeIterator);

	void ForecastTypes(const std::vector<forecast_type>& theTypes);
	void ForecastTypeIterator(const forecast_type_iter& theForecastTypeIterator);

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

	void Create(const grid* baseGrid, bool createDataBackend = false);

	/**
	 * @brief Will reset data backend, ie. create new data that is not attached
	 * to any other info instance. The data content will be the same as in the old
	 * info.
	 *
	 * Will *not* preserve iterator positions.
	 */

	void ReGrid();

	void Producer(long theFmiProducerID);
	void Producer(const producer& theProducer);
	const producer& Producer() const;

	void First();

	/**
	 * @brief Reset all descriptors
	 */

	void Reset();

	/**
	 * @brief Advance dimension iterators in their natural order.
	 *
	 * Location iterator is not advanced
	 */

	bool Next();

	/**
	 * @see iterator#Reset
	 */

	void ResetParam();

	/**
	 * @see iterator#Next
	 */

	bool NextParam();

	/**
	 * @see iterator#First
	 */

	bool FirstParam();

	/**
	 * @brief Set parameter iterator to position indicated by the function argument
	 * @see iterator#Set
	 */

	bool Param(const param& theRequiredParam);
	void ParamIndex(size_t theParamIndex);
	size_t ParamIndex() const;
	param Param() const;
	const param& PeekParam(size_t theIndex) const;
	void SetParam(const param& theParam);

	size_t SizeParams() const;

	/**
	 * @see iterator#Reset
	 */

	void ResetLevel();

	/**
	 * @see iterator#Next
	 */

	bool NextLevel();

	/**
	 * @see iterator#Previous
	 */

	bool PreviousLevel();

	/**
	 * @see iterator#First
	 */

	bool FirstLevel();

	/**
	 * @see iterator#Last
	 */

	bool LastLevel();
	/**
	 * @brief Set level iterator to position indicated by the function argument
	 * @see iterator#Set
	 */

	bool Level(const level& theLevel);
	void LevelIndex(size_t theLevelIndex);
	size_t LevelIndex() const;
	level Level() const;
	const level& PeekLevel(size_t theIndex) const;
	void SetLevel(const level& theLevel);

	HPLevelOrder LevelOrder() const;
	void LevelOrder(HPLevelOrder levelOrder);

	size_t SizeLevels() const;

	/**
	 * @see iterator#Reset
	 */

	void ResetTime();

	/**
	 * @see iterator#Next
	 */

	bool NextTime();

	/**
	 * @see iterator#Previous
	 */

	bool PreviousTime();

	/**
	 * @see iterator#First
	 */

	bool FirstTime();

	/**
	 * @see iterator#Last
	 */

	bool LastTime();

	/**
	 * @brief Set time iterator to position indicated by the function argument
	 * @see iterator#Set
	 */

	bool Time(const forecast_time& theTime);
	void TimeIndex(size_t theTimeIndex);
	size_t TimeIndex() const;
	forecast_time Time() const;
	const forecast_time& PeekTime(size_t theIndex) const;
	void SetTime(const forecast_time& theTime);

	size_t SizeTimes() const;

	/**
	 * @brief Set location iterator to given index value. No limit-checking is made.
	 */

	void LocationIndex(size_t theLocationIndex);
	size_t LocationIndex() const;
	void ResetLocation();
	bool NextLocation();
	bool FirstLocation();
	bool PreviousLocation();
	bool LastLocation();
	size_t LocationIndex();

	size_t SizeLocations() const;

	bool NextForecastType();
	void ResetForecastType();
	bool FirstForecastType();
	size_t ForecastTypeIndex() const;
	size_t SizeForecastTypes() const;
	bool ForecastType(const forecast_type& theType);
	forecast_type ForecastType() const;

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
	 * @return Current data matrix
	 */

	grid* Grid() const;

	/**
	 * @brief Return data matrix from the given time/level/param indexes
	 *
	 * @note Function argument order is important!
	 *
	 * @return Data matrix pointed by the given function arguments.
	 */

	grid* Grid(size_t timeIndex, size_t levelIndex, size_t paramIndex) const;  // Always this order

	/**
	 * @brief Replace current grid with the function argument
	 * @param d shared pointer to a grid instance
	 */

	void Grid(std::shared_ptr<grid> d);

	/**
	 * @brief Shortcut to get the current data matrix
	 * @return Current data matrix
	 */

	matrix<double>& Data();

	/**
	 * @brief Return size of meta matrix. Is the same as times*params*levels.
	 *
	 */

	size_t DimensionSize() const;

	/**
	 * @brief Set the data value pointed by the iterators with a new one
	 */

	void Value(double theValue);

	/**
	 * @return Data value pointed by the iterators
	 */

	double Value() const;

#ifdef HAVE_CUDA

	/**
	 * @brief Stupify this info to a C-style struct
	 *
	 * @return
	 */

	info_simple* ToSimple() const;

#endif

	const std::vector<std::shared_ptr<grid>>& Dimensions() const;

	/**
	 * @brief Clear info contents and iterators
	 *
	 * Does not free memory explicitly.
	 */

	void Clear();

   protected:
	std::unique_ptr<grid> itsBaseGrid;  //!< grid information from json. used as a template, never to store data

   private:
	void Init();

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
	size_t Index() const;

	HPLevelOrder itsLevelOrder;

	level_iter itsLevelIterator;
	time_iter itsTimeIterator;
	param_iter itsParamIterator;
	forecast_type_iter itsForecastTypeIterator;

	std::vector<std::shared_ptr<grid>> itsDimensions;

	logger itsLogger;

	producer itsProducer;

	size_t itsLocationIndex;

#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsLevelOrder), CEREAL_NVP(itsLevelIterator), CEREAL_NVP(itsTimeIterator),
		   CEREAL_NVP(itsParamIterator), CEREAL_NVP(itsForecastTypeIterator), CEREAL_NVP(itsDimensions),
		   CEREAL_NVP(itsBaseGrid), CEREAL_NVP(itsLogger), CEREAL_NVP(itsProducer), CEREAL_NVP(itsLocationIndex));
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const info& ob) { return ob.Write(file); }
inline size_t himan::info::Index(size_t forecastTypeIndex, size_t timeIndex, size_t levelIndex, size_t paramIndex) const
{
	assert(forecastTypeIndex != kIteratorResetValue);
	assert(timeIndex != kIteratorResetValue);
	assert(levelIndex != kIteratorResetValue);
	assert(paramIndex != kIteratorResetValue);

	return (paramIndex * itsForecastTypeIterator.Size() * itsTimeIterator.Size() * itsLevelIterator.Size() +
	        levelIndex * itsForecastTypeIterator.Size() * itsTimeIterator.Size() +
	        timeIndex * itsForecastTypeIterator.Size() + forecastTypeIndex);
}

inline size_t himan::info::Index() const { return Index(ForecastTypeIndex(), TimeIndex(), LevelIndex(), ParamIndex()); }
inline grid* info::Grid() const
{
	assert(itsDimensions.size());
	return itsDimensions[Index()].get();
}

inline grid* info::Grid(size_t timeIndex, size_t levelIndex, size_t paramIndex) const
{
	assert(itsDimensions.size());
	return itsDimensions[Index(ForecastTypeIndex(), timeIndex, levelIndex, paramIndex)].get();
}

inline void info::Value(double theValue) { Grid()->Data().Set(itsLocationIndex, theValue); }
inline double info::Value() const { return Grid()->Data().At(itsLocationIndex); }
typedef std::shared_ptr<info> info_t;

}  // namespace himan

#endif /* INFO_H */
