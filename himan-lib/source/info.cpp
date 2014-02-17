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

info::~info() {}

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
    itsLevelOrder = other.itsLevelOrder;

	itsBottomLeft = other.itsBottomLeft;
	itsTopRight = other.itsTopRight;
	itsSouthPole = other.itsSouthPole;

	itsUVRelativeToGrid = other.itsUVRelativeToGrid;
	itsNi = other.itsNi;
	itsNj = other.itsNj;

	itsDi = other.itsDi;
	itsDj = other.itsDj;

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
    itsLevelOrder = kTopToBottom;

//    itsAB = std::vector<double>;

    itsBottomLeft = point(kHPMissingValue, kHPMissingValue);
    itsTopRight = point(kHPMissingValue, kHPMissingValue);
    itsSouthPole = point(kHPMissingValue, kHPMissingValue);

    itsOrientation = kHPMissingValue;
    itsStepSizeOverOneByte = false;
    itsUVRelativeToGrid = false;

    itsNi = 0;
    itsNj = 0;

    itsDi = kHPMissingValue;
    itsDj = kHPMissingValue;
}

std::ostream& info::Write(std::ostream& file) const
{

    file << "<" << ClassName() << ">" << endl;

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

	for (size_t i = 0; i < itsDimensionMatrix->Size(); i++)
	{
		file << *itsDimensionMatrix->At(i);
	}
	
    return file;
}


void info::Create()
{
    itsDimensionMatrix = shared_ptr<matrix_t> (new matrix_t(itsTimeIterator->Size(), itsLevelIterator->Size(), itsParamIterator->Size()));

    Reset();

	// Disallow Create() to be called if info is not originated from a configuration file

	assert(itsScanningMode != kUnknownScanningMode);
	assert(itsProjection != kUnknownProjection);
    assert(itsLevelOrder != kUnknownLevelOrder);

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

            	if (itsDi != kHPMissingValue && itsDj != kHPMissingValue)
            	{
            		Grid()->Di(itsDi);
            		Grid()->Dj(itsDj);
            	}
            }
        }
    }

	First();
}

void info::ReGrid()
{
	auto newDimensionMatrix = make_shared<matrix_t> (itsTimeIterator->Size(), itsLevelIterator->Size(), itsParamIterator->Size());

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
				assert(Grid());

            	auto newGrid = make_shared<grid> (*Grid());
            	if (itsDi != kHPMissingValue && itsDj != kHPMissingValue)
            	{
            		newGrid->Di(itsDi);
            		newGrid->Dj(itsDj);
            	}

				newDimensionMatrix->Set(TimeIndex(), LevelIndex(), ParamIndex(), newGrid);

            }
        }
    }

	itsDimensionMatrix = newDimensionMatrix;
	First(); // "Factory setting"
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

	First();
}

void info::Merge(shared_ptr<info> otherInfo)
{

    Reset();

	otherInfo->Reset();

	// X = time
	// Y = level
	// Z = param
	
	while (otherInfo->NextTime())
	{
		if (itsTimeIterator->Add(otherInfo->Time())) // no duplicates
		{
			itsDimensionMatrix->SizeX(itsDimensionMatrix->SizeX()+1);
		}

		Time(otherInfo->Time());

		otherInfo->ResetLevel();

		while (otherInfo->NextLevel())
		{
			if (itsLevelIterator->Add(otherInfo->Level())) // no duplicates
			{
				itsDimensionMatrix->SizeY(itsDimensionMatrix->SizeY()+1);
			}

			Level(otherInfo->Level());

			otherInfo->ResetParam();

			while (otherInfo->NextParam())
			{
				if (itsParamIterator->Add(otherInfo->Param())) // no duplicates
				{
					itsDimensionMatrix->SizeZ(itsDimensionMatrix->SizeZ()+1);
				}

				Param(otherInfo->Param());

				Grid(make_shared<grid> (*otherInfo->Grid()));
			}
		}
	}
}


void info::Merge(vector<shared_ptr<info>>& otherInfos)
{

	for (size_t i = 0; i < otherInfos.size(); i++)
	{
		Merge(otherInfos[i]);
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

param& info::PeekParam(size_t theIndex) const
{
	return itsParamIterator->At(theIndex);
}

HPLevelOrder info::LevelOrder() const
{
	return itsLevelOrder;
}
void info::LevelOrder(HPLevelOrder levelOrder)
{
    itsLevelOrder = levelOrder;
}

bool info::NextLevel()
{
    if (itsLevelOrder == kBottomToTop)
        return itsLevelIterator->Previous();
    else
        return itsLevelIterator->Next();
}

bool info::PreviousLevel()
{
    if (itsLevelOrder == kBottomToTop)
        return itsLevelIterator->Next();
    else
        return itsLevelIterator->Previous();
}

bool info::LastLevel()
{
    if (itsLevelOrder == kBottomToTop)
        return itsLevelIterator->First();
    else
        return itsLevelIterator->Last();
}

void info::First()
{
    FirstLevel();
    FirstParam();
    FirstTime();
    FirstLocation();
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
    if (itsLevelOrder == kBottomToTop)
        return itsLevelIterator->Last();
    else
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

level& info::PeekLevel(size_t theIndex) const
{
	return itsLevelIterator->At(theIndex);
}

bool info::NextTime()
{
    return itsTimeIterator->Next();
}

bool info::PreviousTime()
{
    return itsTimeIterator->Previous();
}

bool info::LastTime()
{
    return itsTimeIterator->Last();
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

forecast_time& info::PeekTime(size_t theIndex) const
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

bool info::PreviousLocation()
{
    
    size_t locationSize = Grid()->Data()->Size();

    if (itsLocationIndex == kIteratorResetValue)
    {
        itsLocationIndex = (locationSize == 0) ? 0 : locationSize - 1;   // ResetLocation() has been called before this function
    }

    else
    {
        if (itsLocationIndex == 0)
        {
            itsLocationIndex = (locationSize == 0) ? 0 : locationSize - 1;
            return false;
        }
        itsLocationIndex--;
    }

    return true;

}

bool info::LastLocation()
{
    itsLocationIndex = Grid()->Data()->Size() - 1;

    return true;
}

void info::ResetLocation()
{
    itsLocationIndex = kIteratorResetValue;
}

bool info::FirstLocation()
{
    ResetLocation();

    return NextLocation();
}

size_t info::LocationIndex() const
{
    return itsLocationIndex;
}

void info::LocationIndex(size_t theLocationIndex)
{
    itsLocationIndex = theLocationIndex;
}

size_t info::LocationIndex()
{
    return itsLocationIndex;
}

size_t info::SizeLocations() const
{
	return Data()->Size();
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
	assert(Grid()->Data());
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

HPProjectionType info::Projection() const
{
	// unsafe: will crash if grid is not set?
	return Grid()->Projection();
}

size_t info::DimensionSize() const
{
	return itsDimensionMatrix->Size();
}

void info::ReplaceParam(const param& theParam)
{
	param& p = itsParamIterator->At();

	p = theParam;
}

#ifdef HAVE_CUDA

info_simple* info::ToSimple() const
{
	info_simple* ret = new info_simple();

	ret->size_x = Data()->SizeX();
	ret->size_y = Data()->SizeY();

	ret->di = Grid()->Di();
	ret->dj = Grid()->Dj();

	ret->first_lat = Grid()->FirstGridPoint().Y();
	ret->first_lon = Grid()->FirstGridPoint().X();

	ret->south_pole_lat = Grid()->SouthPole().Y();
	ret->south_pole_lon = Grid()->SouthPole().X();

	if (Grid()->ScanningMode() == kTopLeft)
	{
		ret->j_scans_positive = false;
	}
	else if (Grid()->ScanningMode() != kBottomLeft)
	{
		throw runtime_error(ClassName() + ": Invalid scanning mode for Cuda: " + string(HPScanningModeToString.at(Grid()->ScanningMode())));
	}

	if (Grid()->IsPackedData())
	{

		/*
		 * If grid has packed data, shallow-copy a pointer to that data to 'ret'.
		 * Also allocate page-locked memory for the unpacked data.
		 */

		assert(Grid()->PackedData()->ClassName() == "simple_packed");
		
		shared_ptr<simple_packed> v = dynamic_pointer_cast<simple_packed> (Grid()->PackedData());

		ret->packed_values = v.get();
		ret->create();

	}
	else
	{
		// Shallow-copy pointer to unpacked data. Note: this is not page-locked.

		ret->values = const_cast<double*> (Data()->ValuesAsPOD());
		ret->is_page_locked = false;
	}

	return ret;
}

#endif

const shared_ptr<const matrix_t> info::Dimensions() const
{
	return itsDimensionMatrix;
}
