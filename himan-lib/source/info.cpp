/*
 * info.cpp
 *
 *  Created on: Nov 22, 2012
 *      Author: partio
 */

#include "info.h"
#include <limits> // for std::numeric_limits<size_t>::max();
#include <boost/lexical_cast.hpp>
#include "plugin_factory.h"
#include "logger_factory.h"

#define NEWBASE_INTERPOLATION

#ifdef NEWBASE_INTERPOLATION
#include <NFmiLatLonArea.h>
#include <NFmiRotatedLatLonArea.h>
#include <NFmiStereographicArea.h>
#endif

#define HIMAN_AUXILIARY_INCLUDE

#include "neons.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan;

const size_t kMAX_SIZE_T = std::numeric_limits<size_t>::max();

info::info()
{
	Init();
	itsLogger = logger_factory::Instance()->GetLog("info");

	itsDataMatrix = shared_ptr<matrix_t> (new matrix_t());
}

info::~info()
{
}

shared_ptr<info> info::Clone() const
{

	shared_ptr<info> clone = shared_ptr<info> (new info());

	clone->Projection(itsProjection);
	clone->Orientation(itsOrientation);

	clone->BottomLeftLatitude(itsBottomLeftLatitude);
	clone->BottomLeftLongitude(itsBottomLeftLongitude);
	clone->TopRightLatitude(itsTopRightLatitude);
	clone->TopRightLongitude(itsTopRightLongitude);

	clone->Data(itsDataMatrix);

	clone->Params(itsParams);

	clone->Levels(itsLevels);

	clone->Times(itsTimes);

	clone->Producer(itsProducer);

	clone->OriginDateTime(itsOriginDateTime.String("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S");

	return clone;

}

void info::Init()
{

	itsProjection = kUnknownProjection;

	itsBottomLeftLatitude = kHPMissingFloat;
	itsBottomLeftLongitude = kHPMissingFloat;
	itsTopRightLatitude = kHPMissingFloat;
	itsTopRightLongitude = kHPMissingFloat;
	itsOrientation = kHPMissingFloat;

	itsProducer = kHPMissingInt;

	itsTimeIndex = kMAX_SIZE_T;
	itsLevelIndex = kMAX_SIZE_T;
	itsParamIndex = kMAX_SIZE_T;
	itsLocationIndex = kMAX_SIZE_T;

}

std::ostream& info::Write(std::ostream& file) const
{

	file << "<" << ClassName() << " " << Version() << ">" << endl;

	file << "__itsProjection__ " << itsProjection << endl;
	file << "__itsBottomLeftLongitude__ " << itsBottomLeftLongitude << endl;
	file << "__itsBottomLeftLatitude__ " << itsBottomLeftLatitude << endl;
	file << "__itsTopRightLongitude__ " << itsTopRightLongitude << endl;
	file << "__itsTopRightLongitude__ " << itsTopRightLatitude << endl;
	file << "__itsOrientation__ " << itsOrientation << endl;

	file << "__itsOriginDateTime__ " << OriginDateTime().String() << endl;

	file << "__itsProducer__" << itsProducer << endl;

	if (itsParams.size())
	{
		for (size_t i = 0; i < itsParams.size(); i++)
		{
			file << *itsParams[i];
		}
	}
	else
	{
		file << "__itsParam__ __no-param__" << endl;
	}

	if (itsLevels.size())
	{
		for (size_t i = 0; i < itsLevels.size(); i++)
		{
			file << *itsLevels[i];
		}
	}
	else
	{
		file << "__itsLevel__ __no-level__" << endl;
	}

	if (itsTimes.size())
	{
		for (size_t i = 0; i < itsTimes.size(); i++)
		{
			file << *itsTimes[i];
		}
	}
	else
	{
		file << "__itsTime__ __no-time__" << endl;
	}

	return file;
}


bool info::Create()
{

	itsDataMatrix = shared_ptr<matrix_t> (new matrix_t(itsTimes.size(), itsLevels.size(), itsParams.size()));

	Reset();

	while (NextTime())
	{
		ResetLevel();

		while (NextLevel())
		{
			ResetParam();

			while (NextParam())
				// Create empty placeholders
			{
				itsDataMatrix->Set(CurrentIndex(), shared_ptr<d_matrix_t> (new d_matrix_t(0, 0)));
			}
		}
	}

	return true;

}

HPProjectionType info::Projection() const
{
	return itsProjection;
}

void info::Projection(HPProjectionType theProjection)
{
	itsProjection = theProjection;
}

double info::BottomLeftLatitude() const
{
	return itsBottomLeftLatitude;
}
double info::BottomLeftLongitude() const
{
	return itsBottomLeftLongitude;
}
double info::TopRightLongitude() const
{
	return itsTopRightLongitude;
}
double info::TopRightLatitude() const
{
	return itsTopRightLatitude;
}

void info::BottomLeftLatitude(double theBottomLeftLatitude)
{
	itsBottomLeftLatitude = theBottomLeftLatitude;
}

void info::BottomLeftLongitude(double theBottomLeftLongitude)
{
	itsBottomLeftLongitude = theBottomLeftLongitude;
}
void info::TopRightLatitude(double theTopRightLatitude)
{
	itsTopRightLatitude = theTopRightLatitude;
}
void info::TopRightLongitude(double theTopRightLongitude)
{
	itsTopRightLongitude = theTopRightLongitude;
}

double info::Orientation() const
{
	return itsOrientation;
}

void info::Orientation(double theOrientation)
{
	itsOrientation = theOrientation;
}

unsigned int info::Producer() const
{
	return itsProducer;
}

void info::Producer(unsigned int theProducer)
{
	itsProducer = theProducer;
}

vector<shared_ptr<param> > info::Params() const
{
	return itsParams;
}

void info::Params(vector<shared_ptr<param> > theParams)
{
	itsParams = theParams;
}

vector<shared_ptr<level> > info::Levels() const
{
	return itsLevels;
}

void info::Levels(vector<shared_ptr<level> > theLevels)
{
	itsLevels = theLevels;
}

vector<shared_ptr<forecast_time> > info::Times() const
{
	return itsTimes;
}

void info::Times(vector<shared_ptr<forecast_time> > theTimes)
{
	itsTimes = theTimes;
}

raw_time info::OriginDateTime() const
{
	return itsOriginDateTime;
}

void info::OriginDateTime(const string& theOriginDateTime, const string& theTimeMask)
{
	itsOriginDateTime = raw_time(theOriginDateTime, theTimeMask);
}

bool info::Param(const param& theRequestedParam)
{

	for (size_t i = 0; i < itsParams.size(); i++)
	{
		if (itsParams[i]->Name() == theRequestedParam.Name())
		{
			return true;
		}
	}

	return false;

}

bool info::NextParam()
{

	if (itsParamIndex == kMAX_SIZE_T)
	{
		itsParamIndex = 0;    // ResetParam() has been called before this function
	}

	else
	{
		itsParamIndex++;
	}

	if (itsParamIndex >= itsParams.size())
	{
		itsParamIndex = itsParams.size() == 0 ? 0 : itsParams.size() - 1;
		return false;
	}

	return true;

}

void info::ResetParam()
{
	itsParamIndex = kMAX_SIZE_T;
}

bool info::FirstParam()
{
	ResetParam();

	return NextParam();
}

void info::ParamIndex(size_t theParamIndex)
{
	itsParamIndex = theParamIndex;
}

void info::Param(shared_ptr<const param> theParam)
{

	for (size_t i = 0; i < itsParams.size(); i++)
	{
		if (itsParams[i] == theParam)
		{
			itsParamIndex = i;
		}

	}
}

shared_ptr<param> info::Param() const
{

	if (itsParamIndex != kMAX_SIZE_T && itsParamIndex < itsParams.size())
	{
		return itsParams[itsParamIndex];
	}

	throw runtime_error(ClassName() + ": Invalid param index value: " + boost::lexical_cast<string> (itsParamIndex));
}

bool info::NextLevel()
{

	if (itsLevelIndex == kMAX_SIZE_T)
	{
		itsLevelIndex = 0;    // ResetLevel() has been called before this function
	}

	else
	{
		itsLevelIndex++;
	}

	if (itsLevelIndex >= itsLevels.size())
	{
		itsLevelIndex = itsLevels.size() == 0 ? 0 : itsLevels.size() - 1;

		return false;
	}

	return true;

}

void info::Reset()
{
	ResetLevel();
	ResetParam();
	ResetTime();
	ResetLocation();
}

void info::ResetLevel()
{
	itsLevelIndex = kMAX_SIZE_T;
}

bool info::FirstLevel()
{
	ResetLevel();

	return NextLevel();
}

void info::LevelIndex(size_t theLevelIndex)
{
	itsLevelIndex = theLevelIndex;
}

void info::Level(shared_ptr<const level> theLevel)
{

	for (size_t i = 0; i < itsLevels.size(); i++)
	{
		if (itsLevels[i] == theLevel)
		{
			itsLevelIndex = i;
		}
	}

}

shared_ptr<level> info::Level() const
{

	if (itsLevelIndex != kMAX_SIZE_T && itsLevelIndex < itsLevels.size())
	{
		return itsLevels[itsLevelIndex];
	}

	throw runtime_error(ClassName() + ": Invalid level index value: " + boost::lexical_cast<string> (itsLevelIndex));
}

bool info::NextTime()
{
	if (itsTimeIndex == kMAX_SIZE_T)
	{
		itsTimeIndex = 0;    // ResetTime() has been called before this function
	}

	else
	{
		itsTimeIndex++;
	}

	if (itsTimeIndex >= itsTimes.size())
	{
		itsTimeIndex = (itsTimes.size() == 0) ? 0 : itsTimes.size() - 1;

		return false;
	}

	return true;

}

void info::ResetTime()
{
	itsTimeIndex = kMAX_SIZE_T;
}

bool info::FirstTime()
{
	ResetTime();

	return NextTime();
}

void info::TimeIndex(size_t theTimeIndex)
{
	itsTimeIndex = theTimeIndex;
}

void info::Time(shared_ptr<const forecast_time> theTime)
{

	for (size_t i = 0; i < itsTimes.size(); i++)
	{
		if (itsTimes[i] == theTime)
		{
			itsTimeIndex = i;
		}
	}
}

shared_ptr<forecast_time> info::Time() const
{

	if (itsTimeIndex != kMAX_SIZE_T && itsTimeIndex < itsTimes.size())
	{
		return itsTimes[itsTimeIndex];
	}

	throw runtime_error(ClassName() + ": Invalid time index value: " + boost::lexical_cast<string> (itsTimeIndex));
}

bool info::NextLocation()
{
	if (itsLocationIndex == kMAX_SIZE_T)
	{
		itsLocationIndex = 0;    // ResetLocation() has been called before this function
	}

	else
	{
		itsLocationIndex++;
	}

	size_t locationSize = itsDataMatrix->At(CurrentIndex())->Size();

	if (itsLocationIndex >= locationSize)
	{
		itsLocationIndex = (locationSize == 0) ? 0 : locationSize - 1;

		return false;
	}

	return true;

}

void info::ResetLocation()
{
	itsLocationIndex = kMAX_SIZE_T;
}

bool info::FirstLocation()
{
	ResetLocation();

	return NextTime();
}

size_t info::CurrentIndex() const
{
	return (itsParamIndex * itsLevels.size() * itsTimes.size() + itsLevelIndex * itsTimes.size() + itsTimeIndex);
}

shared_ptr<d_matrix_t> info::Data() const
{
	return itsDataMatrix->At(CurrentIndex());
}

shared_ptr<d_matrix_t> info::Data(size_t timeIndex, size_t levelIndex, size_t paramIndex) const
{
	return itsDataMatrix->At(timeIndex, levelIndex, paramIndex);
}

void info::Data(shared_ptr<matrix_t> m)
{
	itsDataMatrix = m;
}

void info::Data(shared_ptr<d_matrix_t> d)
{
	itsDataMatrix->At(CurrentIndex()) = d;
}

bool info::Value(double theValue)
{
	itsDataMatrix->At(CurrentIndex())->Set(itsLocationIndex, theValue) ;

	return true;
}

double info::Value() const
{
	return itsDataMatrix->At(CurrentIndex())->At(itsLocationIndex);
}

size_t info::Ni() const
{
	return itsDataMatrix->At(CurrentIndex())->SizeX();
}

size_t info::Nj() const
{
	return itsDataMatrix->At(CurrentIndex())->SizeY();
}

bool info::GridAndAreaEquals(std::shared_ptr<const info> other) const
{

	if (itsBottomLeftLatitude != other->BottomLeftLatitude())
	{
		return false;
	}

	if (itsBottomLeftLongitude != other->BottomLeftLongitude())
	{
		return false;
	}

	if (itsTopRightLatitude != other->TopRightLatitude())
	{
		return false;
	}

	if (itsTopRightLongitude != other->TopRightLongitude())
	{
		return false;
	}

	if (itsProjection != other->Projection())
	{
		return false;
	}

	if (itsOrientation != other->Orientation())
	{
		return false;
	}

	if (Ni() != other->Ni())
	{
		return false;
	}

	if (Nj() != other->Nj())
	{
		return false;
	}

	return true;

}

#ifdef NEWBASE_INTERPOLATION

shared_ptr<NFmiGrid> info::ToNewbaseGrid() const
{

	NFmiArea* theArea = 0;

	switch (itsProjection)
	{
		case kLatLonProjection:
			{
				theArea = new NFmiLatLonArea(NFmiPoint(itsBottomLeftLongitude, itsBottomLeftLatitude),
				                             NFmiPoint(itsTopRightLongitude, itsTopRightLatitude));

				break;
			}

		case kRotatedLatLonProjection:
			{
				theArea = new NFmiRotatedLatLonArea(NFmiPoint(itsBottomLeftLongitude, itsBottomLeftLatitude),
				                                    NFmiPoint(itsTopRightLongitude, itsTopRightLatitude),
				                                    NFmiPoint(0., -30.) // south pole location
				                                   );
				break;
			}

		case kStereographicProjection:
			{
				theArea = new NFmiStereographicArea(NFmiPoint(itsBottomLeftLongitude, itsBottomLeftLatitude),
				                                    NFmiPoint(itsTopRightLongitude, itsTopRightLatitude),
				                                    itsOrientation);
				break;

			}

		default:
			throw runtime_error(ClassName() + ": No supported projection found");

			break;
	}

	shared_ptr<NFmiGrid> theGrid (new NFmiGrid(theArea, Ni(), Nj()));

	size_t dataSize = itsDataMatrix->At(CurrentIndex())->Size();

	if (dataSize)   // Do we have data
	{

		NFmiDataPool thePool;

		float* arr = new float[dataSize];

		// convert double array to float

		for (unsigned int i = 0; i < dataSize; i++)
		{
			arr[i] = static_cast<float> (itsDataMatrix->At(CurrentIndex())->At(i));
		}

		if (!thePool.Init(dataSize, arr))
		{
			throw runtime_error("DataPool init failed");
		}

		if (!theGrid->Init(&thePool))
		{
			throw runtime_error("Grid data init failed");
		}

		delete [] arr;
	}

	delete theArea;

	return theGrid;

}

#endif

