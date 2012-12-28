/*
 * info.h
 *
 *  Created on: Nov 22, 2012
 *      Author: partio
 *
 * Class purpose is to server as an in-memory storage of data.
 *
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

namespace himan
{

typedef matrix <std::shared_ptr<d_matrix_t> > matrix_t;

class info
{

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

		size_t Ni() const;
		size_t Nj() const;

		double Di() const;
		double Dj() const;

		std::vector<std::shared_ptr<param>> Params() const;
		void Params(std::vector<std::shared_ptr<param> > theParams);

		std::vector<std::shared_ptr<level>> Levels() const;
		void Levels(std::vector<std::shared_ptr<level> > theLevels);

		std::vector<std::shared_ptr<forecast_time>> Times() const;
		void Times(std::vector<std::shared_ptr<forecast_time> > theTimes);

		/**
		 * @brief Initialize data backend with correct number of zero-sized matrices
		 *
		 * Function will create a number of empty (zero-sized) matrices to
		 * hold the data. The number of the matrices depends on the size
		 * of times, params and levels.
		 *
		 * @todo Should return void?
		 * @return Always true
		 *
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

		raw_time OriginDateTime() const;
		void OriginDateTime(const std::string& theOriginDateTime, const std::string& theTimeMask = "%Y-%m-%d %H:%M:%S");

		bool Param(const param& theRequiredParam);

		void Reset();

		void ResetParam();
		bool NextParam();
		bool FirstParam();

		void Param(std::shared_ptr<const param> theActiveParam);
		void ParamIndex(size_t theParamIndex);
		std::shared_ptr<param> Param() const;

		void ResetLevel();
		bool NextLevel();
		bool FirstLevel();

		void Level(std::shared_ptr<const level> theLevel);
		void LevelIndex(size_t theLevelIndex);
		std::shared_ptr<level> Level() const;

		void ResetTime();
		bool NextTime();
		bool FirstTime();

		void Time(std::shared_ptr<const forecast_time> theTime);
		void TimeIndex(size_t theTimeIndex);
		std::shared_ptr<forecast_time> Time() const;

		void LocationIndex(size_t theLocationIndex);
		void ResetLocation();
		bool NextLocation();
		bool FirstLocation();

		std::shared_ptr<d_matrix_t> Data() const;
		std::shared_ptr<d_matrix_t> Data(size_t timeIndex, size_t levelIndex, size_t paramIndex) const; // Always this order

		void Data(std::shared_ptr<d_matrix_t> d);
		void Data(std::shared_ptr<matrix_t> m);

		size_t DataSize()
		{
			return itsDataMatrix->Size();
		}

		bool Value(double theValue);
		double Value() const;

		std::shared_ptr<NFmiGrid> ToNewbaseGrid() const;
		bool GridAndAreaEquals(std::shared_ptr<const info> other) const;

		HPScanningMode ScanningMode() const;
		void ScanningMode(HPScanningMode theScanningMode);

	private:

		size_t CurrentIndex() const;
		void Init();

		HPProjectionType itsProjection;
		double itsBottomLeftLatitude;
		double itsBottomLeftLongitude;
		double itsTopRightLatitude;
		double itsTopRightLongitude;
		double itsOrientation;

		std::vector<std::shared_ptr<param> > itsParams;
		std::vector<std::shared_ptr<level> > itsLevels;
		std::vector<std::shared_ptr<forecast_time> > itsTimes;

		std::shared_ptr<matrix_t> itsDataMatrix;

		std::unique_ptr<logger> itsLogger;

		producer itsProducer;

		raw_time itsOriginDateTime;

		size_t itsParamIndex;
		size_t itsLevelIndex;
		size_t itsTimeIndex;
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

