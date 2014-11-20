/**
 * @file info.cpp
 *
 * @date Nov 22, 2012
 * @author partio
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
	: itsLevelIterator()
	, itsTimeIterator()
	, itsParamIterator()
	, itsDimensions()
{
	Init();
	itsLogger = logger_factory::Instance()->GetLog("info");

}

info::~info() {}

info::info(const info& other)
	// Iterators are COPIED
	: itsLevelIterator(other.itsLevelIterator)
	, itsTimeIterator(other.itsTimeIterator)
	, itsParamIterator(other.itsParamIterator)
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

	// All the grid-instances area shared
	
	itsDimensions = other.itsDimensions;

	itsLocationIndex = other.itsLocationIndex;

	itsProducer = other.itsProducer;

	itsOriginDateTime = other.itsOriginDateTime;

	itsStepSizeOverOneByte = other.itsStepSizeOverOneByte;
	
	itsLogger = logger_factory::Instance()->GetLog("info");
}

void info::Init()
{

	itsProjection = kUnknownProjection;
	itsScanningMode = kUnknownScanningMode;
	itsLevelOrder = kTopToBottom;

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

	file << "__itsLevelOrder__ " << HPLevelOrderToString.at(itsLevelOrder) << endl;

	file << itsProducer;

   	file << itsParamIterator;
   	file << itsLevelIterator;
   	file << itsTimeIterator;

	for (size_t i = 0; i < itsDimensions.size(); i++)
	{
		file << *itsDimensions[i];
	}
	
	return file;
}


void info::Create()
{
	itsDimensions = vector<shared_ptr<grid>> (itsTimeIterator.Size() * itsLevelIterator.Size() * itsParamIterator.Size());

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
				
				Data()->Resize(itsNi,itsNj);

				if (itsDi != kHPMissingValue && itsDj != kHPMissingValue)
				{
					Grid()->Di(itsDi);
					Grid()->Dj(itsDj);
				}

				Data()->MissingValue(kFloatMissing);
				Data()->Fill(kFloatMissing);
			}
		}
	}

	First();
}

void info::ReGrid()
{
	auto newDimensions = vector<shared_ptr<grid>> (itsTimeIterator.Size() * itsLevelIterator.Size() * itsParamIterator.Size());

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

				newDimensions[Index()] = newGrid;

			}
		}
	}

	itsDimensions = move(newDimensions);
	First(); // "Factory setting"
}

void info::Create(const grid* baseGrid)
{

	itsDimensions = vector<shared_ptr<grid>> (itsTimeIterator.Size() * itsLevelIterator.Size() * itsParamIterator.Size());

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
				Grid(make_shared<grid> (*baseGrid));
			}
		}
	}

	First();
}

void info::Merge(shared_ptr<info> otherInfo)
{

	Reset();

	otherInfo->ResetTime();

	// X = time
	// Y = level
	// Z = param

	while (otherInfo->NextTime())
	{
		if (itsTimeIterator.Add(otherInfo->Time())) // no duplicates
		{
			ReIndex(SizeTimes()-1,SizeLevels(),SizeParams());
		}

		bool ret = Time(otherInfo->Time());

		if (!ret)
		{
			itsLogger->Fatal("Unable to set time, merge failed");
			exit(1);
		}

		otherInfo->ResetLevel();

		while (otherInfo->NextLevel())
		{
			if (itsLevelIterator.Add(otherInfo->Level())) // no duplicates
			{
				ReIndex(SizeTimes(),SizeLevels()-1,SizeParams());
			}

			ret = Level(otherInfo->Level());

			if (!ret)
			{
				itsLogger->Fatal("Unable to set level, merge failed");
				exit(1);
			}

			otherInfo->ResetParam();

			while (otherInfo->NextParam())
			{
				if (itsParamIterator.Add(otherInfo->Param())) // no duplicates
				{
					ReIndex(SizeTimes(),SizeLevels(),SizeParams()-1);
				}

				ret = Param(otherInfo->Param());

				if (!ret)
				{
					itsLogger->Fatal("Unable to set param, merge failed");
					exit(1);
				}

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
	itsParamIterator = theParamIterator;
}

void info::Params(const vector<param>& theParams)
{
	itsParamIterator = param_iter(theParams);
}

void info::LevelIterator(const level_iter& theLevelIterator)
{
	itsLevelIterator = theLevelIterator;
}

void info::Levels(const vector<level>& theLevels)
{
	itsLevelIterator = level_iter(theLevels);
}

void info::TimeIterator(const time_iter& theTimeIterator)
{
	itsTimeIterator = theTimeIterator;
}

void info::Times(const vector<forecast_time>& theTimes)
{
	itsTimeIterator = time_iter(theTimes);
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
	return itsParamIterator.Set(theRequestedParam);
}

bool info::NextParam()
{
	return itsParamIterator.Next();
}

void info::ResetParam()
{
	itsParamIterator.Reset();
}

bool info::FirstParam()
{
	return itsParamIterator.First();
}

size_t info::ParamIndex() const
{
	return itsParamIterator.Index();
}

void info::ParamIndex(size_t theParamIndex)
{
	itsParamIterator.Set(theParamIndex);
}

const param& info::Param() const
{
	return itsParamIterator.At();
}

size_t info::SizeParams() const
{
	return itsParamIterator.Size();
}

const param& info::PeekParam(size_t theIndex) const
{
	return itsParamIterator.At(theIndex);
}

void info::SetParam(const param& theParam)
{
	itsParamIterator.Replace(theParam);
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
	{
		return itsLevelIterator.Previous();
	}
	else
	{
		return itsLevelIterator.Next();
	}
}

bool info::PreviousLevel()
{
	if (itsLevelOrder == kBottomToTop)
	{
		return itsLevelIterator.Next();
	}
	else
	{
		return itsLevelIterator.Previous();
	}
}

bool info::LastLevel()
{
	if (itsLevelOrder == kBottomToTop)
	{
		return itsLevelIterator.First();
	}
	else
	{
		return itsLevelIterator.Last();
	}
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
	itsLevelIterator.Reset();
}

bool info::FirstLevel()
{
	if (itsLevelOrder == kBottomToTop)
	{
		return itsLevelIterator.Last();
	}
	else
	{
		return itsLevelIterator.First();
	}
}

size_t info::LevelIndex() const
{
	return itsLevelIterator.Index();
}

void info::LevelIndex(size_t theLevelIndex)
{
	itsLevelIterator.Set(theLevelIndex);
}

bool info::Level(const level& theLevel)
{
	return itsLevelIterator.Set(theLevel);
}

const level& info::Level() const
{
	return itsLevelIterator.At();
}

size_t info::SizeLevels() const
{
	return itsLevelIterator.Size();
}

const level& info::PeekLevel(size_t theIndex) const
{
	return itsLevelIterator.At(theIndex);
}

void info::SetLevel(const level& theLevel)
{
	itsLevelIterator.Replace(theLevel);
}

bool info::NextTime()
{
	return itsTimeIterator.Next();
}

bool info::PreviousTime()
{
	return itsTimeIterator.Previous();
}

bool info::LastTime()
{
	return itsTimeIterator.Last();
}

void info::ResetTime()
{
	itsTimeIterator.Reset();
}

bool info::FirstTime()
{
	return itsTimeIterator.First();
}

size_t info::TimeIndex() const
{
	return itsTimeIterator.Index();
}

void info::TimeIndex(size_t theTimeIndex)
{
	itsTimeIterator.Set(theTimeIndex);
}

bool info::Time(const forecast_time& theTime)
{
	return itsTimeIterator.Set(theTime);
}

const forecast_time& info::Time() const
{
	return itsTimeIterator.At();
}

size_t info::SizeTimes() const
{
	return itsTimeIterator.Size();
}

const forecast_time& info::PeekTime(size_t theIndex) const
{
	return itsTimeIterator.At(theIndex);
}

void info::SetTime(const forecast_time& theTime)
{
	itsTimeIterator.Replace(theTime);
}

bool info::NextLocation()
{
	if (itsLocationIndex == kIteratorResetValue)
	{
		itsLocationIndex = 0;	// ResetLocation() has been called before this function
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

bool info::PreviousLocation()
{
	
	size_t locationSize = Data()->Size();

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
	itsLocationIndex = Data()->Size() - 1;

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

grid* info::Grid() const
{
	//assert(itsDimensions->At(TimeIndex(), LevelIndex(), ParamIndex()));
	return itsDimensions[Index()].get();
}

grid* info::Grid(size_t timeIndex, size_t levelIndex, size_t paramIndex) const
{
	//assert(itsDimensions->At(timeIndex, levelIndex, paramIndex));
	return itsDimensions[Index(timeIndex, levelIndex, paramIndex)].get();
}

unpacked* info::Data() const
{
	assert(Grid());
	return Grid()->Data();
}

void info::Grid(shared_ptr<grid> d)
{
	itsDimensions[Index()] = d;
}

bool info::Value(double theValue)
{
	return Grid()->Data()->Set(itsLocationIndex, theValue);
}

double info::Value() const
{
	return Grid()->Data()->At(itsLocationIndex);
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
	return Grid()->Projection();
}

size_t info::DimensionSize() const
{
	return itsDimensions.size();
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
		
		ret->packed_values = reinterpret_cast<simple_packed*> (Grid()->PackedData());

	}

	// Reserve a place for the unpacked data
	ret->values = const_cast<double*> (Data()->ValuesAsPOD());
	
	return ret;
}

#endif

const vector<shared_ptr<grid>>& info::Dimensions() const
{
	return itsDimensions;
}

void info::ReIndex(size_t oldXSize, size_t oldYSize, size_t oldZSize)
{
	vector<shared_ptr<grid>> d (SizeTimes() * SizeLevels() * SizeParams());

	for (size_t x = 0; x < oldXSize; x++)
	{
		for (size_t y = 0; y < oldYSize; y++)
		{
			for (size_t z = 0; z < oldZSize; z++)
			{
				d[Index(x, y, z)] = itsDimensions[z * oldXSize * oldYSize + y * oldXSize + x];
				
			}
		}
	}

	itsDimensions = d;
}

point info::LatLon() const
{
	assert(Grid()->Projection() == kLatLonProjection || Grid()->Projection() == kRotatedLatLonProjection);

	if (itsLocationIndex == kIteratorResetValue)
	{
		itsLogger->Error("Location iterator position is not set");
		return point();
	}

	double j;
	point firstPoint = Grid()->FirstGridPoint();

	if (Grid()->ScanningMode() == kBottomLeft) //opts.j_scans_positive)
	{
		j = floor(static_cast<double> (itsLocationIndex / Data()->SizeX()));
	}
	else if (Grid()->ScanningMode() == kTopLeft)
	{
		j = Grid()->Nj() - floor(static_cast<double> (itsLocationIndex / Data()->SizeX()));
	}
	else
	{
		throw runtime_error("Unsupported projection: " + string(HPScanningModeToString.at(Grid()->ScanningMode())));
	}

	double i = itsLocationIndex - j * Grid()->Ni();

	return point(firstPoint.X() + i * Grid()->Di(), firstPoint.Y() + j * Grid()->Dj());
}
