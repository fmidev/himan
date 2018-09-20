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

info::info(const vector<forecast_type>& ftypes, const vector<forecast_time>& times, const vector<level>& levels,
           const vector<param>& params)
    : info()
{
	ForecastTypes(ftypes);
	Times(times);
	Levels(levels);
	Params(params);

	itsDimensions.resize(SizeForecastTypes() * SizeTimes() * SizeLevels() * SizeParams());

	FirstParam();
	FirstTime();
	FirstLevel();
	FirstForecastType();
}

info::info(const forecast_type& ftype, const forecast_time& time, const level& lvl, const param& par)
    : info(vector<forecast_type>({ftype}), vector<forecast_time>({time}), vector<level>({lvl}), vector<param>({par}))
{
}

info::~info()
{
}
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
		if (itsDimensions[i]->grid)
		{
			file << *itsDimensions[i]->grid;
		}
		file << itsDimensions[i]->data;
	}

	return file;
}

void info::Regrid(const vector<param>& newParams)
{
	vector<shared_ptr<base>> newDimensions(itsForecastTypeIterator.Size() * itsTimeIterator.Size() *
	                                       itsLevelIterator.Size() * newParams.size());

	FirstForecastType();
	FirstTime();
	FirstLevel();
	ResetParam();

	while (Next())
	{
		if (IsValidGrid() == false)
		{
			continue;
		}

		size_t newI = (ParamIndex() * SizeForecastTypes() * SizeTimes() * SizeLevels() +
		               LevelIndex() * SizeForecastTypes() * SizeTimes() + TimeIndex() * SizeForecastTypes() +
		               ForecastTypeIndex());

		newDimensions[newI] = make_shared<base>(shared_ptr<grid>(Grid()->Clone()), Data());
	}

	itsDimensions = move(newDimensions);
	First();  // "Factory setting"
}

void info::Regrid(const vector<level>& newLevels)
{
	vector<shared_ptr<base>> newDimensions(itsForecastTypeIterator.Size() * itsTimeIterator.Size() * newLevels.size() *
	                                       itsParamIterator.Size());

	FirstForecastType();
	FirstTime();
	FirstLevel();
	ResetParam();

	while (Next())
	{
		if (IsValidGrid() == false)
		{
			continue;
		}
		size_t newI = (ParamIndex() * SizeForecastTypes() * SizeTimes() * newLevels.size() +
		               LevelIndex() * SizeForecastTypes() * SizeTimes() + TimeIndex() * SizeForecastTypes() +
		               ForecastTypeIndex());

		newDimensions[newI] = make_shared<base>(shared_ptr<grid>(Grid()->Clone()), Data());
	}

	itsDimensions = move(newDimensions);
	First();  // "Factory setting"
}

void info::Create(shared_ptr<base> baseGrid, const param& par, const level& lev, bool createDataBackend)
{
	ASSERT(baseGrid);

	if (itsDimensions.size() == 0)
	{
		itsDimensions.resize(itsForecastTypeIterator.Size() * itsTimeIterator.Size() * itsLevelIterator.Size() *
		                     itsParamIterator.Size());
	}

	FirstForecastType();
	FirstTime();
	FirstLevel();
	ResetParam();

	while (Next())
	{
		if (Level() == lev && Param() == par)
		{
			auto g = shared_ptr<grid>(baseGrid->grid->Clone());
			auto d = matrix<double>(baseGrid->data);

			auto b = make_shared<base>(g, d);

			Base(b);

			if (baseGrid->grid->Class() == kRegularGrid)
			{
				const regular_grid* regGrid(dynamic_cast<const regular_grid*>(baseGrid->grid.get()));
				if (createDataBackend)
				{
					Data().Resize(regGrid->Ni(), regGrid->Nj());
				}
			}
			else if (baseGrid->grid->Class() == kIrregularGrid)
			{
				Data().Resize(Grid()->Size(), 1, 1);
			}
			else
			{
				throw runtime_error(ClassName() + ": Invalid grid type");
			}
		}
	}
}

void info::Create(shared_ptr<base> baseGrid, bool createDataBackend)
{
	ASSERT(baseGrid);

	itsDimensions.resize(itsForecastTypeIterator.Size() * itsTimeIterator.Size() *
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
					auto g = shared_ptr<grid>(baseGrid->grid->Clone());
					auto d = matrix<double>(baseGrid->data);

					auto b = make_shared<base>(g, d);

					Base(b);

					if (baseGrid->grid->Class() == kRegularGrid)
					{
						if (createDataBackend)
						{
							const regular_grid* regGrid(dynamic_cast<const regular_grid*>(baseGrid->grid.get()));
							Data().Resize(regGrid->Ni(), regGrid->Nj());
						}
					}
					else if (baseGrid->grid->Class() == kIrregularGrid)
					{
						Data().Resize(Grid()->Size(), 1, 1);
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
			himan::Abort();
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
				himan::Abort();
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
					himan::Abort();
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
						himan::Abort();
					}

					Base(make_shared<base>(shared_ptr<grid>(otherInfo->Grid()->Clone()), otherInfo->Data()));
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
	// If dimensions are reduced we are not regridding, that would mean that we'd have
	// to (most likely) remove some data which might not be what the caller wanted.
	// We can revisit this functionality later.
	if (!itsDimensions.empty() && itsParamIterator.Size() && itsParamIterator.Size() < theParams.size())
	{
		Regrid(theParams);
	}

	itsParamIterator = param_iter(theParams);
}
void info::LevelIterator(const level_iter& theLevelIterator)
{
	itsLevelIterator = theLevelIterator;
}
void info::Levels(const vector<level>& theLevels)
{
	if (!itsDimensions.empty() && itsLevelIterator.Size() && itsLevelIterator.Size() < theLevels.size())
	{
		Regrid(theLevels);
	}

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
param info::Param() const
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

void info::ResetLevel()
{
	itsLevelIterator.Reset();
}
bool info::FirstLevel()
{
	ASSERT(itsLevelOrder != kUnknownLevelOrder);
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
level info::Level() const
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
forecast_time info::Time() const
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
bool info::NextForecastType()
{
	return itsForecastTypeIterator.Next();
}
size_t info::SizeForecastTypes() const
{
	return itsForecastTypeIterator.Size();
}
void info::ResetForecastType()
{
	itsForecastTypeIterator.Reset();
}
size_t info::ForecastTypeIndex() const
{
	return itsForecastTypeIterator.Index();
}
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
	return itsDimensions[Index()]->data.Size();
}
matrix<double>& info::Data()
{
	return itsDimensions[Index()]->data;
}
shared_ptr<grid> info::Grid() const
{
	ASSERT(itsDimensions.size());
	return itsDimensions[Index()]->grid;
}
std::shared_ptr<base> info::Base()
{
	return itsDimensions[Index()];
}
void info::Base(shared_ptr<base> b)
{
	ASSERT(itsDimensions.size() > Index());
	itsDimensions[Index()] = b;
}
shared_ptr<packed_data> info::PackedData() const
{
	return itsDimensions[Index()]->pdata;
}
size_t info::DimensionSize() const
{
	return itsDimensions.size();
}
#ifdef HAVE_CUDA

info_simple* info::ToSimple() const
{
	info_simple* ret = new info_simple();

	ret->size_x = itsDimensions[Index()]->data.SizeX();
	ret->size_y = itsDimensions[Index()]->data.SizeY();

	if (Grid()->Class() == kRegularGrid)
	{
		ret->di = dynamic_pointer_cast<regular_grid>(Grid())->Di();
		ret->dj = dynamic_pointer_cast<regular_grid>(Grid())->Dj();
	}
	else
	{
		ret->di = kHPMissingValue;
		ret->dj = kHPMissingValue;
	}

	ret->first_lat = Grid()->FirstPoint().Y();
	ret->first_lon = Grid()->FirstPoint().X();

	if (Grid()->Type() == kRotatedLatitudeLongitude)
	{
		ret->south_pole_lat = dynamic_pointer_cast<rotated_latitude_longitude_grid>(Grid())->SouthPole().Y();
		ret->south_pole_lon = dynamic_pointer_cast<rotated_latitude_longitude_grid>(Grid())->SouthPole().X();
	}
	else if (Grid()->Type() == kStereographic)
	{
		ret->orientation = dynamic_pointer_cast<stereographic_grid>(Grid())->Orientation();
	}
	else if (Grid()->Type() == kLambertConformalConic)
	{
		auto llc = dynamic_pointer_cast<lambert_conformal_grid>(Grid());
		ret->orientation = llc->Orientation();
		ret->latin1 = llc->StandardParallel1();
		ret->latin2 = llc->StandardParallel2();
	}

	ret->interpolation = Param().InterpolationMethod();

	if (Grid()->Class() == kRegularGrid)
	{
		auto gr = dynamic_pointer_cast<regular_grid>(Grid());

		if (gr->ScanningMode() == kTopLeft)
		{
			ret->j_scans_positive = false;
		}
		else if (gr->ScanningMode() != kBottomLeft)
		{
			throw runtime_error(ClassName() + ": Invalid scanning mode for Cuda: " +
			                    string(HPScanningModeToString.at(gr->ScanningMode())));
		}
	}

	ret->projection = Grid()->Type();

	if (PackedData()->HasData())
	{
		/*
		 * If info has packed data, shallow-copy a pointer to that data to 'ret'.
		 * Also allocate page-locked memory for the unpacked data.
		 */

		ASSERT(PackedData()->ClassName() == "simple_packed");

		ret->packed_values = reinterpret_cast<simple_packed*>(PackedData().get());
	}

	// Reserve a place for the unpacked data
	ret->values = const_cast<double*>(itsDimensions[Index()]->data.ValuesAsPOD());

	return ret;
}

#endif

vector<shared_ptr<base>> info::Dimensions() const
{
	return itsDimensions;
}
void info::ReIndex(size_t oldForecastTypeSize, size_t oldTimeSize, size_t oldLevelSize, size_t oldParamSize)
{
	vector<shared_ptr<base>> theDimensions(SizeForecastTypes() * SizeTimes() * SizeLevels() * SizeParams());

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

	return dynamic_pointer_cast<point_list>(Grid())->Station(itsLocationIndex);
}

void info::ForecastTypes(const std::vector<forecast_type>& theTypes)
{
	itsForecastTypeIterator = forecast_type_iter(theTypes);
}

void info::ForecastTypeIterator(const forecast_type_iter& theForecastTypeIterator)
{
	itsForecastTypeIterator = theForecastTypeIterator;
}

forecast_type info::ForecastType() const
{
	return itsForecastTypeIterator.At();
}
bool info::ForecastType(const forecast_type& theRequestedType)
{
	return itsForecastTypeIterator.Set(theRequestedType);
}
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

time_iter& info::TimeIterator()
{
	return itsTimeIterator;
}
param_iter& info::ParamIterator()
{
	return itsParamIterator;
}
level_iter& info::LevelIterator()
{
	return itsLevelIterator;
}
forecast_type_iter& info::ForecastTypeIterator()
{
	return itsForecastTypeIterator;
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

bool info::IsValidGrid() const
{
	return (itsDimensions[Index()] != nullptr && Grid());
}

/*
info info::Clone()
{
    auto ret = info(*this);

    ret.FirstForecastType();
    ret.FirstTime();
    ret.FirstLevel();
    ret.ResetParam();

    while (Next())
    {
        const auto g = Grid();

        if (g)
        {
            Grid(shared_ptr<grid>(g->Clone()));
        }
    }

    // Return indices
    ret.ForecastTypeIterator().Set(ForecastTypeIndex());
    ret.TimeIterator().Set(TimeIndex());
    ret.LevelIterator().Set(LevelIndex());
    ret.ParamIterator().Set(ParamIndex());

    return ret;
}
*/

void info::FirstValidGrid()
{
	for (ResetParam(); NextParam();)
	{
		if (IsValidGrid())
		{
			return;
		}
	}

	itsLogger.Fatal("A dimension with no valid infos? Madness!");
	himan::Abort();
}
