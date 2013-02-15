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

info::info()
	: itsLevelIterator(new level_iter())
	, itsTimeIterator(new time_iter())
	, itsParamIterator(new param_iter())
	, itsDimensionMatrix(new matrix_t())
{
    Init();
    itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("info"));

}

info::~info()
{
}

info::info(const info& other)
	// Iterators are COPIED
	: itsLevelIterator(new level_iter(*other.itsLevelIterator))
	, itsTimeIterator(new time_iter(*other.itsTimeIterator))
	, itsParamIterator(new param_iter(*other.itsParamIterator))
{
	/* START GLOBAL CONFIGURATION OPTIONS */

	itsProjection = other.itsProjection;
	itsOrientation = other.itsOrientation;
	itsScanningMode = other.itsScanningMode;

	itsBottomLeft = other.itsBottomLeft;
	itsTopRight = other.itsTopRight;
	itsSouthPole = other.itsSouthPole;

	itsUVRelativeToGrid = other.itsUVRelativeToGrid;
	itsNi = other.itsNi;
	itsNj = other.itsNj;

	/* END GLOBAL CONFIGURATION OPTIONS */

	// Data backend is SHARED
	itsDimensionMatrix = other.itsDimensionMatrix;


	itsLocationIndex = other.itsLocationIndex;

	itsProducer = other.itsProducer;

	itsOriginDateTime = other.itsOriginDateTime;

	itsStepSizeOverOneByte = other.itsStepSizeOverOneByte;

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("info"));
}

void info::Init()
{

    itsProjection = kUnknownProjection;
    itsScanningMode = kUnknownScanningMode;

    itsBottomLeft = point(kHPMissingFloat, kHPMissingFloat);
    itsTopRight = point(kHPMissingFloat, kHPMissingFloat);
    itsSouthPole = point(kHPMissingFloat, kHPMissingFloat);

    itsOrientation = kHPMissingFloat;
    itsStepSizeOverOneByte = false;
    itsUVRelativeToGrid = false;

    itsNi = 0;
    itsNj = 0;

}

std::ostream& info::Write(std::ostream& file) const
{

    file << "<" << ClassName() << " " << Version() << ">" << endl;

    file << "__itsProjection__ " << itsProjection << endl;

    file << itsBottomLeft;
    file << itsTopRight;
    file << itsSouthPole;

    file << "__itsOrientation__ " << itsOrientation << endl;
    file << "__itsUVRelativeToGrid__ " << itsUVRelativeToGrid << endl;

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


void info::Create()
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
            	Grid(shared_ptr<grid> (new grid(itsScanningMode, itsUVRelativeToGrid, itsProjection, itsBottomLeft, itsTopRight, itsSouthPole, itsOrientation)));
            	Grid()->Data()->Resize(itsNi,itsNj);
            }
        }
    }

}

void info::Create(shared_ptr<grid> baseGrid)
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
            	Grid(shared_ptr<grid> (new grid(*baseGrid)));
            }
        }
    }

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

size_t info::SizeParams() const
{
	return itsParamIterator->Size();
}

param& info::PeakParam(size_t theIndex) const
{
	return itsParamIterator->At(theIndex);
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

size_t info::SizeLevels() const
{
	return itsLevelIterator->Size();
}

level& info::PeakLevel(size_t theIndex) const
{
	return itsLevelIterator->At(theIndex);
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

size_t info::SizeTimes() const
{
	return itsTimeIterator->Size();
}

forecast_time& info::PeakTime(size_t theIndex) const
{
	return itsTimeIterator->At(theIndex);
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

bool info::StepSizeOverOneByte() const
{
	return itsStepSizeOverOneByte;
}

void info::StepSizeOverOneByte(bool theStepSizeOverOneByte)
{
	itsStepSizeOverOneByte = theStepSizeOverOneByte;
}

std::vector<plugin_configuration> info::Plugins() const
{
    return itsPlugins;
}

void info::Plugins(const std::vector<plugin_configuration>& thePlugins)
{
    itsPlugins = thePlugins;
}
