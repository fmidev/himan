/**
 * @file info.cpp
 *
 * @date Nov 22, 2012
 * @author partio
 */

#include "info.h"
#include "grid.h"
#include "lambert_conformal_grid.h"
#include "latitude_longitude_grid.h"
#include "point_list.h"
#include "stereographic_grid.h"
#include <limits>  // for std::numeric_limits<size_t>::max();

using namespace std;
using namespace himan;

info::info()
    : itsBaseGrid(),
      itsLevelOrder(kTopToBottom),
      itsLevelIterator(),
      itsTimeIterator(),
      itsParamIterator(),
      itsForecastTypeIterator(),
      itsDimensions(),
      itsLogger(logger("info")),
      itsLocationIndex(kIteratorResetValue)
{
}

info::~info() {}
info::info(const info& other)
    // Iterators are COPIED
    : itsLevelOrder(other.itsLevelOrder),
      itsLevelIterator(other.itsLevelIterator),
      itsTimeIterator(other.itsTimeIterator),
      itsParamIterator(other.itsParamIterator),
      itsForecastTypeIterator(other.itsForecastTypeIterator),
      itsDimensions(other.itsDimensions),
      itsProducer(other.itsProducer),
      itsLocationIndex(other.itsLocationIndex)
{
	if (other.itsBaseGrid)
	{
		itsBaseGrid = unique_ptr<grid>(other.itsBaseGrid->Clone());
	}

	itsLogger = logger("info");
}

std::ostream& info::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << endl;

	file << "__itsLevelOrder__ " << HPLevelOrderToString.at(itsLevelOrder) << endl;

	file << itsProducer;

	file << itsParamIterator;
	file << itsLevelIterator;
	file << itsTimeIterator;
	file << itsForecastTypeIterator;

	for (size_t i = 0; i < itsDimensions.size(); i++)
	{
		if (itsDimensions[i]) file << *itsDimensions[i];
	}

	return file;
}

void info::ReGrid()
{
	auto newDimensions =
	    vector<shared_ptr<grid>>(itsTimeIterator.Size() * itsLevelIterator.Size() * itsParamIterator.Size());

	Reset();

	while (NextForecastType())
	{
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

					newDimensions[Index()] = shared_ptr<grid>(Grid()->Clone());
				}
			}
		}
	}

	itsDimensions = move(newDimensions);
	First();  // "Factory setting"
}

void info::Create(const grid* baseGrid, bool createDataBackend)
{
	assert(baseGrid);

	itsDimensions = vector<shared_ptr<grid>>(itsForecastTypeIterator.Size() * itsTimeIterator.Size() *
	                                         itsLevelIterator.Size() * itsParamIterator.Size());

	ResetForecastType();

	while (NextForecastType())
	{
		ResetTime();

		while (NextTime())
		{
			ResetLevel();

			while (NextLevel())
			{
				ResetParam();

				while (NextParam())
				// Create empty placeholders
				{
					Grid(shared_ptr<grid>(baseGrid->Clone()));

					if (baseGrid->Class() == kRegularGrid)
					{
						if (createDataBackend)
						{
							Grid()->Data().Resize(Grid()->Ni(), Grid()->Nj());
						}
					}
					else if (baseGrid->Class() == kIrregularGrid)
					{
						if (baseGrid->Type() == kReducedGaussian)
						{
							Grid()->Data().Resize(Grid()->Size(), 1, 1);
						}
					}
					else
					{
						throw runtime_error(ClassName() + ": Invalid grid type");
					}
				}
			}
		}
	}

	First();
}

void info::Merge(shared_ptr<info> otherInfo)
{
	Reset();

	otherInfo->ResetForecastType();

	// X = forecast type
	// Y = time
	// Z = level
	// Å = param

	while (otherInfo->NextForecastType())
	{
		if (itsForecastTypeIterator.Add(otherInfo->ForecastType()))  // no duplicates
		{
			ReIndex(SizeForecastTypes() - 1, SizeTimes(), SizeLevels(), SizeParams());
		}

		if (!ForecastType(otherInfo->ForecastType()))
		{
			itsLogger.Fatal("Unable to set forecast type, merge failed");
			abort();
		}

		otherInfo->ResetTime();

		while (otherInfo->NextTime())
		{
			if (itsTimeIterator.Add(otherInfo->Time()))  // no duplicates
			{
				ReIndex(SizeForecastTypes(), SizeTimes() - 1, SizeLevels(), SizeParams());
			}

			if (!Time(otherInfo->Time()))
			{
				itsLogger.Fatal("Unable to set time, merge failed");
				abort();
			}

			otherInfo->ResetLevel();

			while (otherInfo->NextLevel())
			{
				if (itsLevelIterator.Add(otherInfo->Level()))  // no duplicates
				{
					ReIndex(SizeForecastTypes(), SizeTimes(), SizeLevels() - 1, SizeParams());
				}

				if (!Level(otherInfo->Level()))
				{
					itsLogger.Fatal("Unable to set level, merge failed");
					abort();
				}

				otherInfo->ResetParam();

				while (otherInfo->NextParam())
				{
					if (itsParamIterator.Add(otherInfo->Param()))  // no duplicates
					{
						ReIndex(SizeForecastTypes(), SizeTimes(), SizeLevels(), SizeParams() - 1);
					}

					if (!Param(otherInfo->Param()))
					{
						itsLogger.Fatal("Unable to set param, merge failed");
						abort();
					}

					Grid(shared_ptr<grid>(otherInfo->Grid()->Clone()));
				}
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

const producer& info::Producer() const { return itsProducer; }
void info::Producer(long theFmiProducerId) { itsProducer = producer(theFmiProducerId); }
void info::Producer(const producer& theProducer) { itsProducer = theProducer; }
void info::ParamIterator(const param_iter& theParamIterator) { itsParamIterator = theParamIterator; }
void info::Params(const vector<param>& theParams) { itsParamIterator = param_iter(theParams); }
void info::LevelIterator(const level_iter& theLevelIterator) { itsLevelIterator = theLevelIterator; }
void info::Levels(const vector<level>& theLevels) { itsLevelIterator = level_iter(theLevels); }
void info::TimeIterator(const time_iter& theTimeIterator) { itsTimeIterator = theTimeIterator; }
void info::Times(const vector<forecast_time>& theTimes) { itsTimeIterator = time_iter(theTimes); }
bool info::Param(const param& theRequestedParam) { return itsParamIterator.Set(theRequestedParam); }
bool info::NextParam() { return itsParamIterator.Next(); }
void info::ResetParam() { itsParamIterator.Reset(); }
bool info::FirstParam() { return itsParamIterator.First(); }
size_t info::ParamIndex() const { return itsParamIterator.Index(); }
void info::ParamIndex(size_t theParamIndex) { itsParamIterator.Set(theParamIndex); }
param info::Param() const { return itsParamIterator.At(); }
size_t info::SizeParams() const { return itsParamIterator.Size(); }
const param& info::PeekParam(size_t theIndex) const { return itsParamIterator.At(theIndex); }
void info::SetParam(const param& theParam) { itsParamIterator.Replace(theParam); }
HPLevelOrder info::LevelOrder() const { return itsLevelOrder; }
void info::LevelOrder(HPLevelOrder levelOrder) { itsLevelOrder = levelOrder; }
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
	FirstForecastType();
	FirstLocation();
}

void info::Reset()
{
	ResetLevel();
	ResetParam();
	ResetTime();
	ResetLocation();
	ResetForecastType();
}

void info::ResetLevel() { itsLevelIterator.Reset(); }
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

size_t info::LevelIndex() const { return itsLevelIterator.Index(); }
void info::LevelIndex(size_t theLevelIndex) { itsLevelIterator.Set(theLevelIndex); }
bool info::Level(const level& theLevel) { return itsLevelIterator.Set(theLevel); }
level info::Level() const { return itsLevelIterator.At(); }
size_t info::SizeLevels() const { return itsLevelIterator.Size(); }
const level& info::PeekLevel(size_t theIndex) const { return itsLevelIterator.At(theIndex); }
void info::SetLevel(const level& theLevel) { itsLevelIterator.Replace(theLevel); }
bool info::NextTime() { return itsTimeIterator.Next(); }
bool info::PreviousTime() { return itsTimeIterator.Previous(); }
bool info::LastTime() { return itsTimeIterator.Last(); }
void info::ResetTime() { itsTimeIterator.Reset(); }
bool info::FirstTime() { return itsTimeIterator.First(); }
size_t info::TimeIndex() const { return itsTimeIterator.Index(); }
void info::TimeIndex(size_t theTimeIndex) { itsTimeIterator.Set(theTimeIndex); }
bool info::Time(const forecast_time& theTime) { return itsTimeIterator.Set(theTime); }
forecast_time info::Time() const { return itsTimeIterator.At(); }
size_t info::SizeTimes() const { return itsTimeIterator.Size(); }
const forecast_time& info::PeekTime(size_t theIndex) const { return itsTimeIterator.At(theIndex); }
void info::SetTime(const forecast_time& theTime) { itsTimeIterator.Replace(theTime); }
bool info::NextForecastType() { return itsForecastTypeIterator.Next(); }
size_t info::SizeForecastTypes() const { return itsForecastTypeIterator.Size(); }
void info::ResetForecastType() { itsForecastTypeIterator.Reset(); }
size_t info::ForecastTypeIndex() const { return itsForecastTypeIterator.Index(); }
bool info::NextLocation()
{
	if (itsLocationIndex == kIteratorResetValue)
	{
		itsLocationIndex = 0;  // ResetLocation() has been called before this function
	}

	else
	{
		itsLocationIndex++;
	}

	size_t locationSize = Data().Size();

	if (itsLocationIndex >= locationSize)
	{
		itsLocationIndex = (locationSize == 0) ? 0 : locationSize - 1;

		return false;
	}

	return true;
}

bool info::PreviousLocation()
{
	size_t locationSize = Data().Size();

	if (itsLocationIndex == kIteratorResetValue)
	{
		itsLocationIndex =
		    (locationSize == 0) ? 0 : locationSize - 1;  // ResetLocation() has been called before this function
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
	itsLocationIndex = Data().Size() - 1;

	return true;
}

void info::ResetLocation() { itsLocationIndex = kIteratorResetValue; }
bool info::FirstLocation()
{
	ResetLocation();

	return NextLocation();
}

size_t info::LocationIndex() const { return itsLocationIndex; }
void info::LocationIndex(size_t theLocationIndex) { itsLocationIndex = theLocationIndex; }
size_t info::LocationIndex() { return itsLocationIndex; }
size_t info::SizeLocations() const { return Grid()->Data().Size(); }
matrix<double>& info::Data()
{
	assert(Grid());
	return Grid()->Data();
}

void info::Grid(shared_ptr<grid> d)
{
	assert(itsDimensions.size() > Index());
	itsDimensions[Index()] = d;
}

size_t info::DimensionSize() const { return itsDimensions.size(); }
#ifdef HAVE_CUDA

info_simple* info::ToSimple() const
{
	info_simple* ret = new info_simple();

	ret->size_x = Grid()->Data().SizeX();
	ret->size_y = Grid()->Data().SizeY();

	ret->di = Grid()->Di();
	ret->dj = Grid()->Dj();

	ret->first_lat = Grid()->FirstPoint().Y();
	ret->first_lon = Grid()->FirstPoint().X();

	if (Grid()->Type() == kRotatedLatitudeLongitude)
	{
		ret->south_pole_lat = dynamic_cast<rotated_latitude_longitude_grid*>(Grid())->SouthPole().Y();
		ret->south_pole_lon = dynamic_cast<rotated_latitude_longitude_grid*>(Grid())->SouthPole().X();
	}
	else if (Grid()->Type() == kStereographic)
	{
		ret->orientation = dynamic_cast<stereographic_grid*>(Grid())->Orientation();
	}
	else if (Grid()->Type() == kLambertConformalConic)
	{
		ret->orientation = dynamic_cast<lambert_conformal_grid*>(Grid())->Orientation();
		ret->latin1 = dynamic_cast<lambert_conformal_grid*>(Grid())->StandardParallel1();
		ret->latin2 = dynamic_cast<lambert_conformal_grid*>(Grid())->StandardParallel2();
	}

	ret->interpolation = Param().InterpolationMethod();

	if (Grid()->ScanningMode() == kTopLeft)
	{
		ret->j_scans_positive = false;
	}
	else if (Grid()->ScanningMode() != kBottomLeft)
	{
		throw runtime_error(ClassName() + ": Invalid scanning mode for Cuda: " +
		                    string(HPScanningModeToString.at(Grid()->ScanningMode())));
	}

	ret->projection = Grid()->Type();

	if (Grid()->IsPackedData())
	{
		/*
		 * If grid has packed data, shallow-copy a pointer to that data to 'ret'.
		 * Also allocate page-locked memory for the unpacked data.
		 */

		assert(Grid()->PackedData().ClassName() == "simple_packed");

		ret->packed_values = reinterpret_cast<simple_packed*>(&Grid()->PackedData());
	}

	// Reserve a place for the unpacked data
	ret->values = const_cast<double*>(Grid()->Data().ValuesAsPOD());

	return ret;
}

#endif

const vector<shared_ptr<grid>>& info::Dimensions() const { return itsDimensions; }
void info::ReIndex(size_t oldForecastTypeSize, size_t oldTimeSize, size_t oldLevelSize, size_t oldParamSize)
{
	vector<shared_ptr<grid>> theDimensions(SizeForecastTypes() * SizeTimes() * SizeLevels() * SizeParams());

	for (size_t a = 0; a < oldForecastTypeSize; a++)
	{
		for (size_t b = 0; b < oldTimeSize; b++)
		{
			for (size_t c = 0; c < oldLevelSize; c++)
			{
				for (size_t d = 0; d < oldParamSize; d++)
				{
					size_t index = d * oldForecastTypeSize * oldTimeSize * oldLevelSize +
					               c * oldForecastTypeSize * oldTimeSize + b * oldForecastTypeSize + a;

					size_t newIndex = Index(a, b, c, d);
					theDimensions[newIndex] = itsDimensions[index];
				}
			}
		}
	}

	itsDimensions = theDimensions;
}

point info::LatLon() const
{
	if (itsLocationIndex == kIteratorResetValue)
	{
		itsLogger.Error("Location iterator position is not set");
		return point();
	}

	return Grid()->LatLon(itsLocationIndex);
}

station info::Station() const
{
	if (itsLocationIndex == kIteratorResetValue)
	{
		itsLogger.Error("Location iterator position is not set");
		return station();
	}
	else if (Grid()->Class() != kIrregularGrid)
	{
		itsLogger.Error("regular_grid does not hold station information");
		return station();
	}

	return dynamic_cast<point_list*>(Grid())->Station(itsLocationIndex);
}

void info::ForecastTypes(const std::vector<forecast_type>& theTypes)
{
	itsForecastTypeIterator = forecast_type_iter(theTypes);
}

void info::ForecastTypeIterator(const forecast_type_iter& theForecastTypeIterator)
{
	itsForecastTypeIterator = theForecastTypeIterator;
}

forecast_type info::ForecastType() const { return itsForecastTypeIterator.At(); }
bool info::ForecastType(const forecast_type& theRequestedType) { return itsForecastTypeIterator.Set(theRequestedType); }
bool info::FirstForecastType()
{
	ResetForecastType();
	return NextForecastType();
}

void info::Clear()
{
	itsDimensions.clear();

	itsParamIterator.Clear();
	itsLevelIterator.Clear();
	itsTimeIterator.Clear();
	itsForecastTypeIterator.Clear();
}

bool info::Next()
{
	// Innermost

	if (NextParam())
	{
		return true;
	}

	// No more params at this forecast type/level/time combination; rewind param iterator

	FirstParam();

	if (NextLevel())
	{
		return true;
	}

	// No more levels at this forecast type/time combination; rewind level iterator

	FirstLevel();

	if (NextTime())
	{
		return true;
	}

	// No more times at this forecast type; rewind time iterator, level iterator is
	// already at first place

	FirstTime();

	if (NextForecastType())
	{
		return true;
	}

	return false;
}
