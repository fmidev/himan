/*
 * info.h
 *
 *  Created on: Nov 22, 2012
 *      Author: partio
 *
 * Class purpose is to server as an in-memory storage of data.
 *
 * Actual data storage is derived from newbase (QueryInfo).
 *
 */

#ifndef HILPEE_INFO_H
#define HILPEE_INFO_H

#include <NFmiGrid.h>
#include "logger.h"
#include "param.h"
#include "level.h"
#include "forecast_time.h"
#include "matrix.h"

namespace hilpee
{

typedef matrix <std::shared_ptr<d_matrix_t> > matrix_t;

class info
{

	public:

		//friend class configuration;

		info();

		~info();

		std::string ClassName() const
		{
			return "hilpee::info";
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

		std::vector<std::shared_ptr<param>> Params() const;
		void Params(std::vector<std::shared_ptr<param>> theParams);

		std::vector<std::shared_ptr<level>> Levels() const;
		void Levels(std::vector<std::shared_ptr<level>> theLevels);

		std::vector<std::shared_ptr<forecast_time>> Times() const;
		void Times(std::vector<std::shared_ptr<forecast_time>> theTimes);

		bool Create();

		std::shared_ptr<info> Clone() const;

		void Producer(unsigned int theProducer);
		unsigned int Producer() const;

		raw_time OriginDateTime() const;
		void OriginDateTime(const std::string& theOriginDateTime, const std::string& theTimeMask = "%Y-%m-%d %H:%M:%S");

		bool Param(const param& theRequiredParam);

		void Reset();

		void ResetParam();
		bool NextParam();
		bool FirstParam();

		void Param(std::shared_ptr<param> theActiveParam);
		void ParamIndex(size_t theParamIndex);
		std::shared_ptr<param> Param() const;

		void ResetLevel();
		bool NextLevel();
		bool FirstLevel();

		void Level(std::shared_ptr<level> theLevel);
		void LevelIndex(size_t theLevelIndex);
		std::shared_ptr<level> Level() const;

		void ResetTime();
		bool NextTime();
		bool FirstTime();

		void Time(std::shared_ptr<forecast_time> theTime);
		void TimeIndex(size_t theTimeIndex);
		std::shared_ptr<forecast_time> Time() const;

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

	private:
		info(const info& theInfo);
		size_t CurrentIndex() const;
		void Init();

		HPProjectionType itsProjection;
		double itsBottomLeftLatitude;
		double itsBottomLeftLongitude;
		double itsTopRightLatitude;
		double itsTopRightLongitude;
		double itsOrientation;

		std::vector<std::shared_ptr<param>> itsParams;
		std::vector<std::shared_ptr<level>> itsLevels;
		std::vector<std::shared_ptr<forecast_time>> itsTimes;

		std::shared_ptr<matrix_t> itsDataMatrix;

		std::unique_ptr<logger> itsLogger;
		unsigned int itsProducer;

		raw_time itsOriginDateTime;

		size_t itsParamIndex;
		size_t itsLevelIndex;
		size_t itsTimeIndex;
		size_t itsLocationIndex;

};

inline
std::ostream& operator<<(std::ostream& file, info& ob)
{
	return ob.Write(file);
}

} // namespace hilpee

#endif /* INFO_H */

