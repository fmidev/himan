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

#define HIMAN_AUXILIARY_INCLUDE

#include "neons.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan;

info::info() : itsLevelIterator(), itsTimeIterator(), itsParamIterator()
{
    Init();
    itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("info"));

    itsDimensionMatrix = shared_ptr<matrix_t> (new matrix_t());
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
    //clone->ScanningMode(itsScanningMode);

    clone->BottomLeft(itsBottomLeft);
    clone->TopRight(itsTopRight);
    clone->SouthPole(itsSouthPole);

    clone->Data(itsDimensionMatrix);

    clone->ParamIterator(*itsParamIterator);
    clone->LevelIterator(*itsLevelIterator);
    clone->TimeIterator(*itsTimeIterator);

    clone->Producer(itsProducer);

    clone->OriginDateTime(itsOriginDateTime.String("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S");

    clone->LocationIndex(itsLocationIndex);

    clone->StepSizeOverOneByte(itsStepSizeOverOneByte);
    return clone;

}

void info::Init()
{

    itsProjection = kUnknownProjection;

    itsBottomLeft = point(kHPMissingFloat, kHPMissingFloat);
    itsTopRight = point(kHPMissingFloat, kHPMissingFloat);
    itsSouthPole = point(kHPMissingFloat, kHPMissingFloat);

    itsOrientation = kHPMissingFloat;
    itsStepSizeOverOneByte = false;
}

std::ostream& info::Write(std::ostream& file) const
{

    file << "<" << ClassName() << " " << Version() << ">" << endl;

    file << "__itsProjection__ " << itsProjection << endl;

    file << itsBottomLeft;
    file << itsTopRight;
    file << itsSouthPole;

    file << "__itsOrientation__ " << itsOrientation << endl;

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


bool info::Create(HPScanningMode theScanningMode, bool theUVRelativeToGrid)
{

    itsDimensionMatrix = shared_ptr<matrix_t> (new matrix_t(itsTimeIterator->Size(), itsLevelIterator->Size(), itsParamIterator->Size()));

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
            	Grid(shared_ptr<grid> (new grid(theScanningMode, theUVRelativeToGrid, itsProjection, itsBottomLeft, itsTopRight, itsSouthPole, itsOrientation)));

//            	Grid(shared_ptr<grid> (new grid()));
  //          	Grid()->ScanningMode(itsScanningMode);
            }
        }
    }

    return true;

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

void info::ParamIterator(const param_iter& theParamIterator)
{
    itsParamIterator = shared_ptr<param_iter> (new param_iter(theParamIterator));
}

void info::Params(const vector<param>& theParams)
{
    itsParamIterator = shared_ptr<param_iter> (new param_iter(theParams));
}

void info::LevelIterator(const level_iter& theLevelIterator)
{
    itsLevelIterator = shared_ptr<level_iter> (new level_iter(theLevelIterator));
}

void info::Levels(const vector<level>& theLevels)
{
    itsLevelIterator = shared_ptr<level_iter> (new level_iter(theLevels));
}

void info::TimeIterator(const time_iter& theTimeIterator)
{
    itsTimeIterator = shared_ptr<time_iter> (new time_iter(theTimeIterator));
}

void info::Times(const vector<forecast_time>& theTimes)
{
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

    size_t locationSize = Grid()->Data()->Size();

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

shared_ptr<grid> info::Grid() const
{
    return itsDimensionMatrix->At(TimeIndex(), LevelIndex(), ParamIndex());
}

shared_ptr<grid> info::Grid(size_t timeIndex, size_t levelIndex, size_t paramIndex) const
{
    return itsDimensionMatrix->At(timeIndex, levelIndex, paramIndex);
}

shared_ptr<d_matrix_t> info::Data() const
{
	return Grid()->Data();
}

void info::Data(shared_ptr<matrix_t> m)
{
    itsDimensionMatrix = m;
}

void info::Grid(shared_ptr<grid> d)
{
    itsDimensionMatrix->At(TimeIndex(), LevelIndex(), ParamIndex()) = d;
}

bool info::Value(double theValue)
{
    return Grid()->Data()->Set(itsLocationIndex, theValue) ;
}

double info::Value() const
{
    return Grid()->Data()->At(itsLocationIndex);
}

size_t info::Ni() const
{
    return Grid()->Data()->SizeX();
}

size_t info::Nj() const
{
    return Grid()->Data()->SizeY();
}

double info::Di() const
{
	return Grid()->Di();
}

double info::Dj() const
{
	return Grid()->Dj();
}

HPProjectionType info::Projection() const
{
	return itsProjection;
}

void info::Projection(HPProjectionType theProjection)
{
	itsProjection = theProjection;
}

point info::BottomLeft() const
{
	return itsBottomLeft;
}

point info::TopRight() const
{
	return itsTopRight;
}

void info::BottomLeft(const point& theBottomLeft)
{
	itsBottomLeft = theBottomLeft;
}

void info::TopRight(const point& theTopRight)
{
	itsTopRight = theTopRight;
}

void info::SouthPole(const point& theSouthPole)
{
    itsSouthPole = theSouthPole;
}

point info::SouthPole() const
{
    return itsSouthPole;
}

double info::Orientation() const
{
    return itsOrientation;
}

void info::Orientation(double theOrientation)
{
    itsOrientation = theOrientation;
}

bool info::StepSizeOverOneByte() const
{
	return itsStepSizeOverOneByte;
}

void info::StepSizeOverOneByte(bool theStepSizeOverOneByte)
{
	itsStepSizeOverOneByte = theStepSizeOverOneByte;
}
