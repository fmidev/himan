/**
 * @file info.h
 *
 * @brief Define metadata structures requred to calculate and store data
 *
 * @date Nov 22, 2012
 *
 * @author partio
 *
 */

#ifndef INFO_H
#define INFO_H

#include <NFmiGrid.h>
#include "logger.h"
#include "param.h"
#include "level.h"
#include "forecast_time.h"
#include "matrix.h"
#include "himan_common.h"
#include "producer.h"

#define NEWBASE_INTERPOLATION

namespace himan
{

typedef matrix <std::shared_ptr<d_matrix_t> > matrix_t;

const size_t kIteratorResetValue = std::numeric_limits<size_t>::max();

class info
{

public:

    /**
     * @class iterator
     *
     * @brief Nested class inside info to provide iterator functions to info class
     *
     */

    template <class T>
    class iterator
    {
    public:

        iterator<T>() {}
        iterator<T>(const std::vector<T>& theElements)
        {
            itsElements = theElements;
            Reset();
        }

        std::string ClassName() const
        {
            return "himan::info::iterator";
        }

        HPVersionNumber Version() const
        {
            return HPVersionNumber(0, 1);
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
         * @brief Advance iterator by one
         *
         * @return boolean if iterator has more elements left, else false
         *
         */

        bool Next()
        {
            if (itsIndex == kIteratorResetValue)
            {
                itsIndex = 0;    // Reset() has been called before this function
            }

            else
            {
                itsIndex++;
            }

            if (itsIndex >= itsElements.size())
            {
                itsIndex = itsElements.size() == 0 ? 0 : itsElements.size() - 1;
                return false;
            }

            return true;
        }

        /**
         * @return Reference to current value or throw exception
         */

        T& At()
        {
            if (itsIndex != kIteratorResetValue && itsIndex < itsElements.size())
            {
                return itsElements[itsIndex];
            }

            throw std::runtime_error(ClassName() + ": Invalid index value: " + boost::lexical_cast<std::string> (itsIndex));

        }

        /**
         * @brief Set iterator to the position indicated by the function argument
         *
         * @return True if value exists, else false
         *
         */

        bool Set(const T theElement)
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

        void Set(size_t theIndex)
        {
            itsIndex = theIndex;
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

        friend std::ostream& operator<<(std::ostream& file, iterator<T> & ob)
        {
            return ob.Write(file);
        }

        /**
         * @brief Write object to stream
         */

        std::ostream& Write(std::ostream& file) const
        {
            file << "<" << ClassName() << " " << Version() << ">" << std::endl;
            file << "__itsIndex__ " << itsIndex << std::endl;
            file << "__itsSize__ " << itsElements.size() << std::endl;

            return file;
        }

    private:
        std::vector<T> itsElements; //<! Vector to hold the elements
        size_t itsIndex; //<! Current index of iterator

    };

    typedef iterator<level> level_iter;
    typedef iterator<param> param_iter;
    typedef iterator<forecast_time> time_iter;

public:

    info();
    ~info();

    info(const info& other) = delete;
    info& operator=(const info& other) = delete;

    std::string ClassName() const
    {
        return "himan::info";
    }

    HPVersionNumber Version() const
    {
        return HPVersionNumber(0, 1);
    }

    std::ostream& Write(std::ostream& file) const;

    /**
     * @return Projection type of this info
     *
     * One info can hold only one type of projection
     */

    HPProjectionType Projection() const;
    void Projection(HPProjectionType theProjection);

    double BottomLeftLatitude() const;
    double BottomLeftLongitude() const;
    double TopRightLongitude() const;
    double TopRightLatitude() const;

    void BottomLeftLatitude(double theBottomLeftLatitude);
    void BottomLeftLongitude(double theBottomLeftLongitude);
    void TopRightLatitude(double theTopRightLatitude);
    void TopRightLongitude(double theTopRightLongitude);

    void Orientation(double theOrientation);
    double Orientation() const;

    double SouthPoleLongitude() const;
    double SouthPoleLatitude() const;

    void SouthPoleLongitude(double theSouthPoleLongitude);
    void SouthPoleLatitude(double theSouthPoleLatitude);

    /**
     * @brief Calculates latitude of first grid point based on the area definition and scanning mode
     * @return Latitude of first grid point
     * @todo How this works with stereographic projection
     */
    double LatitudeOfFirstGridPoint() const;

    /**
	 * @brief Calculates longitude of first grid point based on the area definition and scanning mode
	 * @return Latitude of first grid point
	 * @todo How this works with stereographic projection
	 */

    double LongitudeOfFirstGridPoint() const;

    /**
     * @return Number of point along X axis
     */

    size_t Ni() const;

    /**
     * @return Number of point along Y axis
     */

    size_t Nj() const;

    /**
     * @return Distance between two points in X axis in degrees
     */

    double Di() const;

    /**
     * @return Distance between two points in Y axis in degrees
     */

    double Dj() const;

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

    /**
     * @brief Initialize data backend with correct number of zero-sized matrices
     *
     * Function will create a number of empty (zero-sized) matrices to
     * hold the data. The number of the matrices depends on the size
     * of times, params and levels.
     *
     * @todo Should return void?
     * @return Always true
     */

    bool Create();

    /**
     * @brief Clone info while preserving its data backend
     *
     * Function will return a new info (wrapped with shared_ptr) than
     * has the same data backend matrix as the original one. This means
     * that multiple threads can access the same data with different
     * infos ( --> descriptor positions ). Clone will have the same
     * descriptor positions.
     *
     * @return New info with access to same data backend
     *
     */

    std::shared_ptr<info> Clone() const;

    void Producer(long theFmiProducerID);
    void Producer(const producer& theProducer);
    const producer& Producer() const;

    /**
     * @brief Return info-wide origin time if it exists
     * @return Origin time
     */

    raw_time OriginDateTime() const;
    void OriginDateTime(const std::string& theOriginDateTime, const std::string& theTimeMask = "%Y-%m-%d %H:%M:%S");

    /**
     * @brief Reset all descriptors
     */

    void Reset();

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
    param& Param() const;

    /**
     * @see iterator#Reset
     */

    void ResetLevel();

    /**
     * @see iterator#Next
     */

    bool NextLevel();

    /**
     * @see iterator#First
     */

    bool FirstLevel();

    /**
     * @brief Set level iterator to position indicated by the function argument
     * @see iterator#Set
     */

    bool Level(const level& theLevel);
    void LevelIndex(size_t theLevelIndex);
    size_t LevelIndex() const;
    level& Level() const;

    /**
     * @see iterator#Reset
     */

    void ResetTime();

    /**
     * @see iterator#Next
     */

    bool NextTime();

    /**
     * @see iterator#First
     */

    bool FirstTime();

    /**
     * @brief Set time iterator to position indicated by the function argument
     * @see iterator#Set
     */

    bool Time(const forecast_time& theTime);
    void TimeIndex(size_t theTimeIndex);
    size_t TimeIndex() const;
    forecast_time& Time() const;

    /**
     * @brief Set location iterator to given index value. No limit-checking is made.
     */

    void LocationIndex(size_t theLocationIndex);
    void ResetLocation();
    bool NextLocation();
    bool FirstLocation();

    /**
     * @return Current data matrix
     */

    std::shared_ptr<d_matrix_t> Data() const;

    /**
     * @brief Return data matrix from the given time/level/param indexes
     *
     * @note Function argument order is important!
     *
     * @return Data matrix pointed by the given function arguments.
     */

    std::shared_ptr<d_matrix_t> Data(size_t timeIndex, size_t levelIndex, size_t paramIndex) const; // Always this order

    /**
     * @brief Replace current data matrix with the function argument
     * @param d shared pointer to a data matrix
     */

    void Data(std::shared_ptr<d_matrix_t> d);

    /**
     * @brief Replace whole meta matrix with a new one
     * @param m shared pointer to n meta matrix
     */
    void Data(std::shared_ptr<matrix_t> m);

    /**
     * @brief Return size of meta matrix. Is the same as times*params*levels.
     *
     */

    size_t DataSize() const
    {
        return itsDataMatrix->Size();
    }

    /**
     * @brief Set the data value pointed by the iterators with a new one
     * @return True if assignment was succesfull
     */

    bool Value(double theValue);

    /**
     * @return Data value pointed by the iterators
     */

    double Value() const;

#ifdef NEWBASE_INTERPOLATION

    /**
     * @brief Create a newbase grid (NFmiGrid) from current data
     * @todo Should return raw pointer ?
     */

    std::shared_ptr<NFmiGrid> ToNewbaseGrid() const;

    #endif

    /**
     * @brief Check if grid and area are equal
     */

    bool GridAndAreaEquals(std::shared_ptr<const info> other) const;

    HPScanningMode ScanningMode() const;
    void ScanningMode(HPScanningMode theScanningMode);

private:

    //size_t CurrentIndex() const;
    void Init();

    HPProjectionType itsProjection;
    double itsBottomLeftLatitude;
    double itsBottomLeftLongitude;
    double itsTopRightLatitude;
    double itsTopRightLongitude;

    double itsSouthPoleLatitude;
    double itsSouthPoleLongitude;

    double itsOrientation;

    std::shared_ptr<level_iter> itsLevelIterator;
    std::shared_ptr<time_iter> itsTimeIterator;
    std::shared_ptr<param_iter> itsParamIterator;

    std::shared_ptr<matrix_t> itsDataMatrix;

    std::unique_ptr<logger> itsLogger;

    producer itsProducer;

    raw_time itsOriginDateTime;

    size_t itsLocationIndex;

    HPScanningMode itsScanningMode; //<! When data is read from files, we need to know what is the scanning mode

};

inline
std::ostream& operator<<(std::ostream& file, info& ob)
{
    return ob.Write(file);
}

} // namespace himan

#endif /* INFO_H */

