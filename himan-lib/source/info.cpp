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

info::info()
{
    Init();
    itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("info"));

    itsDataMatrix = shared_ptr<matrix_t> (new matrix_t());
    itsTimeIterator = shared_ptr<time_iter> (new time_iter());
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

    clone->ParamIterator(*itsParamIterator);
    clone->LevelIterator(*itsLevelIterator);
    clone->TimeIterator(*itsTimeIterator);

    clone->Producer(itsProducer);

    clone->OriginDateTime(itsOriginDateTime.String("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S");

    /*
    clone->ParamIndex(itsParamIndex);
    clone->TimeIndex(itsTimeIndex);
    clone->LevelIndex(itsLevelIndex);
    */
    clone->LocationIndex(itsLocationIndex);

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

    itsScanningMode = kTopLeft;

}

std::ostream& info::Write(std::ostream& file) const
{

    file << "<" << ClassName() << " " << Version() << ">" << endl;

    file << "__itsProjection__ " << itsProjection << endl;
    file << "__itsBottomLeftLongitude__ " << itsBottomLeftLongitude << endl;
    file << "__itsBottomLeftLatitude__ " << itsBottomLeftLatitude << endl;
    file << "__itsTopRightLongitude__ " << itsTopRightLongitude << endl;
    file << "__itsTopRightLatitude__ " << itsTopRightLatitude << endl;
    file << "__itsOrientation__ " << itsOrientation << endl;

    file << "__itsOriginDateTime__ " << OriginDateTime().String() << endl;

    file << "__itsProducer__ " << itsProducer << endl;

    file << itsParamIterator << endl;
    file << itsLevelIterator << endl;
    file << itsTimeIterator << endl;

    /*
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
    */
    return file;
}


bool info::Create()
{

    itsDataMatrix = shared_ptr<matrix_t> (new matrix_t(itsTimeIterator->Size(), itsLevelIterator->Size(), itsParamIterator->Size()));

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

const producer& info::Producer() const
{
    return itsProducer;
}

void info::Producer(long theFmiProducerId)
{
    itsProducer = producer(theFmiProducerId);
}


void info::Producer(const producer& theProducer)
{
    itsProducer = theProducer;
}
/*
vector<shared_ptr<param> > info::Params() const
{
	return itsParams;
}
*/
void info::ParamIterator(const param_iter& theParamIterator)
{
    itsParamIterator = shared_ptr<param_iter> (new param_iter(theParamIterator));
}

void info::Params(const vector<param>& theParams)
{
    itsParamIterator = shared_ptr<param_iter> (new param_iter(theParams));
}
/*
vector<shared_ptr<level> > info::Levels() const
{
	return itsLevels;
}
*/

void info::LevelIterator(const level_iter& theLevelIterator)
{
    itsLevelIterator = shared_ptr<level_iter> (new level_iter(theLevelIterator));
}

void info::Levels(const vector<level>& theLevels)
{
    //itsLevels = theLevels;
    itsLevelIterator = shared_ptr<level_iter> (new level_iter(theLevels));
}
/*
vector<shared_ptr<forecast_time> > info::Times() const
{
	return itsTimes;
}*/

void info::TimeIterator(const time_iter& theTimeIterator)
{
    itsTimeIterator = shared_ptr<time_iter> (new time_iter(theTimeIterator));
}

void info::Times(const vector<forecast_time>& theTimes)
{
    //itsTimes = theTimes;
    itsTimeIterator = shared_ptr<time_iter> (new time_iter(theTimes));
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
    return itsParamIterator->Set(theRequestedParam);
}

bool info::NextParam()
{
    return itsParamIterator->Next();
}

void info::ResetParam()
{
    itsParamIterator->Reset();
}

bool info::FirstParam()
{
    return itsParamIterator->First();
}

size_t info::ParamIndex() const
{
    return itsParamIterator->Index();
}

void info::ParamIndex(size_t theParamIndex)
{
    itsParamIterator->Set(theParamIndex);
}

param& info::Param() const
{
    return itsParamIterator->At();
}

bool info::NextLevel()
{
    return itsLevelIterator->Next();
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
    itsLevelIterator->Reset();
}

bool info::FirstLevel()
{
    return itsLevelIterator->First();
}

size_t info::LevelIndex() const
{
    return itsLevelIterator->Index();
}

void info::LevelIndex(size_t theLevelIndex)
{
    itsLevelIterator->Set(theLevelIndex);
}

bool info::Level(const level& theLevel)
{
    return itsLevelIterator->Set(theLevel);
}

level& info::Level() const
{
    return itsLevelIterator->At();
}

bool info::NextTime()
{
    return itsTimeIterator->Next();
}

void info::ResetTime()
{
    itsTimeIterator->Reset();
}

bool info::FirstTime()
{
    return itsTimeIterator->First();
}

size_t info::TimeIndex() const
{
    return itsTimeIterator->Index();
}

void info::TimeIndex(size_t theTimeIndex)
{
    itsTimeIterator->Set(theTimeIndex);
}

bool info::Time(const forecast_time& theTime)
{
    return itsTimeIterator->Set(theTime);
}

forecast_time& info::Time() const
{
    return itsTimeIterator->At();
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

void info::LocationIndex(size_t theLocationIndex)
{
    itsLocationIndex = theLocationIndex;
}

size_t info::CurrentIndex() const
{
    return (itsParamIterator->Index() * itsLevelIterator->Size() * itsTimeIterator->Size() + itsLevelIterator->Index() * itsTimeIterator->Size() + itsTimeIterator->Index());
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

double info::Di() const
{
    return abs((itsBottomLeftLongitude - itsTopRightLongitude) / Ni());
}

double info::Dj() const
{
    return abs((itsBottomLeftLatitude - itsTopRightLatitude) / Nj());
}

HPScanningMode info::ScanningMode() const
{
    return itsScanningMode;
}

void info::ScanningMode(HPScanningMode theScanningMode)
{
    itsScanningMode = theScanningMode;
}

bool info::GridAndAreaEquals(shared_ptr<const info> other) const
{

    // TODO: a clever way to compare if two areas are equal
    // for example, coordinates could be backwards (-90 to 90 vs 90 to -90)

    if (itsBottomLeftLatitude != other->BottomLeftLatitude())
    {
        itsLogger->Trace("BottomLeftLatitudes aren't the same");
        return false;
    }

    if (itsBottomLeftLongitude != other->BottomLeftLongitude())
    {
        itsLogger->Trace("BottomLeftLongitude aren't the same");
        return false;
    }

    if (itsTopRightLatitude != other->TopRightLatitude())
    {
        itsLogger->Trace("TopRightLatitudes aren't the same");
        return false;
    }

    if (itsTopRightLongitude != other->TopRightLongitude())
    {
        cout << itsTopRightLongitude << " != " <<  other->TopRightLongitude() << endl;

        itsLogger->Trace("TopRightLongitudes aren't the same");
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

    FmiInterpolationMethod interp = kLinearly;
    FmiDirection dir = static_cast<FmiDirection> (itsScanningMode);

    NFmiArea* theArea = 0;

    // Newbase does not understand grib2 longitude coordinates

    double bottomLeftLongitude = itsBottomLeftLongitude;
    double topRightLongitude = itsTopRightLongitude;

    if (bottomLeftLongitude > 180 || topRightLongitude > 180)
    {
        bottomLeftLongitude -= 180;
        topRightLongitude -= 180;
    }

    switch (itsProjection)
    {
    case kLatLonProjection:
    {
        theArea = new NFmiLatLonArea(NFmiPoint(bottomLeftLongitude, itsBottomLeftLatitude),
                                     NFmiPoint(topRightLongitude, itsTopRightLatitude));

        break;
    }

    case kRotatedLatLonProjection:
    {
        theArea = new NFmiRotatedLatLonArea(NFmiPoint(bottomLeftLongitude, itsBottomLeftLatitude),
                                            NFmiPoint(topRightLongitude, itsTopRightLatitude),
                                            NFmiPoint(0., -30.) // south pole location
                                           );
        break;
    }

    case kStereographicProjection:
    {
        theArea = new NFmiStereographicArea(NFmiPoint(bottomLeftLongitude, itsBottomLeftLatitude),
                                            NFmiPoint(topRightLongitude, itsTopRightLatitude),
                                            itsOrientation);
        break;

    }

    default:
        throw runtime_error(ClassName() + ": No supported projection found");

        break;
    }

    shared_ptr<NFmiGrid> theGrid (new NFmiGrid(theArea, Ni(), Nj(), dir, interp));

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

