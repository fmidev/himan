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

info::info() : itsLevelIterator(), itsTimeIterator(), itsParamIterator()
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
    clone->ScanningMode(itsScanningMode);

    clone->BottomLeftLatitude(itsBottomLeftLatitude);
    clone->BottomLeftLongitude(itsBottomLeftLongitude);
    clone->TopRightLatitude(itsTopRightLatitude);
    clone->TopRightLongitude(itsTopRightLongitude);

    clone->SouthPoleLatitude(itsSouthPoleLatitude);
    clone->SouthPoleLongitude(itsSouthPoleLongitude);

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

    itsSouthPoleLatitude = kHPMissingFloat;
    itsSouthPoleLongitude = kHPMissingFloat;

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
    file << "__itsSouthPoleLongitude__ " << itsSouthPoleLongitude << endl;
    file << "__itsSouthPoleLatitude__ " << itsSouthPoleLatitude << endl;

    file << "__itsScanningMode__ " << itsScanningMode << endl;

    file << itsProducer;

    if (itsParamIterator)
    {
    	file << *itsParamIterator;
    }

    if (itsLevelIterator)
    {
    	file << *itsLevelIterator;
    }

    if (itsTimeIterator)
    {
    	file << *itsTimeIterator;
    }

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
            	Data(shared_ptr<d_matrix_t> (new d_matrix_t(0, 0)));
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

void info::SouthPoleLatitude(double theSouthPoleLatitude)
{
    itsSouthPoleLatitude = theSouthPoleLatitude;
}

double info::SouthPoleLatitude() const
{
    return itsSouthPoleLatitude;
}

void info::SouthPoleLongitude(double theSouthPoleLongitude)
{
    itsSouthPoleLongitude = theSouthPoleLongitude;
}

double info::SouthPoleLongitude() const
{
    return itsSouthPoleLongitude;
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
    if (itsLocationIndex == kIteratorResetValue)
    {
        itsLocationIndex = 0;    // ResetLocation() has been called before this function
    }

    else
    {
        itsLocationIndex++;
    }

    size_t locationSize = Data()->Size();

    if (itsLocationIndex >= locationSize)
    {
        itsLocationIndex = (locationSize == 0) ? 0 : locationSize - 1;

        return false;
    }

    return true;

}

void info::ResetLocation()
{
    itsLocationIndex = kIteratorResetValue;
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

shared_ptr<d_matrix_t> info::Data() const
{
    return itsDataMatrix->At(TimeIndex(), LevelIndex(), ParamIndex());
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
    itsDataMatrix->At(TimeIndex(), LevelIndex(), ParamIndex()) = d;
}

bool info::Value(double theValue)
{
    return Data()->Set(itsLocationIndex, theValue) ;
}

double info::Value() const
{
    return Data()->At(itsLocationIndex);
}

size_t info::Ni() const
{
    return Data()->SizeX();
}

size_t info::Nj() const
{
    return Data()->SizeY();
}

double info::Di() const
{
    return abs((itsBottomLeftLongitude - itsTopRightLongitude) / (Ni()-1));
}

double info::Dj() const
{
    return abs((itsBottomLeftLatitude - itsTopRightLatitude) / (Nj()-1));
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

	const float kCoordinateEpsilon = 0.00001;

	/**
	 * @todo A clever way to compare if two areas are equal. For newbase we need bottomleft/topright coordinates
     * but grib gives us just first gridpoints.
	 *
	 */

    if (fabs(itsBottomLeftLatitude - other->BottomLeftLatitude()) > kCoordinateEpsilon)
    {
        itsLogger->Trace("BottomLeftLatitudes don't match: " + boost::lexical_cast<string> (itsBottomLeftLatitude) + " vs " + boost::lexical_cast<string> (other->BottomLeftLatitude()));
        return false;
    }

    if (fabs(itsBottomLeftLongitude - other->BottomLeftLongitude()) > kCoordinateEpsilon)
    {
        itsLogger->Trace("BottomLeftLongitude don't match: " + boost::lexical_cast<string> (itsBottomLeftLongitude) + " vs " + boost::lexical_cast<string> (other->BottomLeftLongitude()));
        return false;
    }

    if (fabs(itsTopRightLatitude - other->TopRightLatitude()) > kCoordinateEpsilon)
    {
        itsLogger->Trace("TopRightLatitudes don't match: " + boost::lexical_cast<string> (itsTopRightLatitude) + " vs " + boost::lexical_cast<string> (other->TopRightLatitude()));
        return false;
    }

    if (fabs(itsTopRightLongitude - other->TopRightLongitude()) > kCoordinateEpsilon)
    {
        itsLogger->Trace("TopRightLongitudes don't match: " + boost::lexical_cast<string> (itsTopRightLongitude) + " vs " + boost::lexical_cast<string> (other->TopRightLongitude()));
        return false;
    }

    if (itsProjection != other->Projection())
    {
        itsLogger->Trace("Projections don't match: " + boost::lexical_cast<string> (itsProjection) + " vs " + boost::lexical_cast<string> (other->Projection()));
        return false;
    }

    if (itsProjection == kRotatedLatLonProjection)
    {
		if (fabs(itsSouthPoleLatitude - other->SouthPoleLatitude()) > kCoordinateEpsilon)
    	{
        	itsLogger->Trace("SouthPoleLatitudes don't match: " + boost::lexical_cast<string> (itsSouthPoleLatitude) + " vs " + boost::lexical_cast<string> (other->SouthPoleLatitude()));
        	return false;
    	}
		if (fabs(itsSouthPoleLongitude - other->SouthPoleLongitude()) > kCoordinateEpsilon)
    	{
        	itsLogger->Trace("SouthPoleLongitudes don't match: " + boost::lexical_cast<string> (itsSouthPoleLongitude) + " vs " + boost::lexical_cast<string> (other->SouthPoleLongitude()));
        	return false;
    	}
    }

    if (itsOrientation != other->Orientation())
    {
    	itsLogger->Trace("Orientations don't match: " + boost::lexical_cast<string> (itsOrientation) + " vs " + boost::lexical_cast<string> (other->Orientation()));
        return false;
    }

    if (Ni() != other->Ni())
    {
    	itsLogger->Trace("Grid X-counts don't match: " + boost::lexical_cast<string> (Ni()) + " vs " + boost::lexical_cast<string> (other->Ni()));
        return false;
    }

    if (Nj() != other->Nj())
    {
    	itsLogger->Trace("Grid Y-counts don't match: " + boost::lexical_cast<string> (Nj()) + " vs " + boost::lexical_cast<string> (other->Nj()));
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
                                            NFmiPoint(itsSouthPoleLongitude, itsSouthPoleLatitude),
                                            NFmiPoint(0.,0.), // default values
                                            NFmiPoint(1.,1.), // default values
                                            true);
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

    size_t dataSize = Data()->Size();

    if (dataSize)   // Do we have data
    {

        NFmiDataPool thePool;

        float* arr = new float[dataSize];

        // convert double array to float

        for (unsigned int i = 0; i < dataSize; i++)
        {
            arr[i] = static_cast<float> (Data()->At(i));
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

