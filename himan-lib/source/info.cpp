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
	Set<forecast_type>(ftypes);
	Set<forecast_time>(times);
	Set<level>(levels);
	Set<param>(params);

	itsDimensions.resize(Size<forecast_type>() * Size<forecast_time>() * Size<level>() * Size<param>());

	First<param>();
	First<forecast_time>();
	First<level>();
	First<forecast_type>();
}

info::info(const forecast_type& ftype, const forecast_time& time, const level& lvl, const param& par)
    : info(vector<forecast_type>({ftype}), vector<forecast_time>({time}), vector<level>({lvl}), vector<param>({par}))
{
}

info::info(const info& other)
    // Iterators are COPIED
    : itsLevelIterator(other.itsLevelIterator),
      itsTimeIterator(other.itsTimeIterator),
      itsParamIterator(other.itsParamIterator),
      itsForecastTypeIterator(other.itsForecastTypeIterator),
      itsDimensions(other.itsDimensions),
      itsProducer(other.itsProducer),
      itsLocationIndex(other.itsLocationIndex)
{
	if (other.itsBaseGrid)
	{
		itsBaseGrid = other.itsBaseGrid->Clone();
	}

	itsLogger = logger("info");
}

std::ostream& info::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << endl;

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

void info::Create(shared_ptr<base> baseGrid, bool createDataBackend)
{
	ASSERT(baseGrid);

	itsDimensions.resize(itsForecastTypeIterator.Size() * itsTimeIterator.Size() * itsLevelIterator.Size() *
	                     itsParamIterator.Size());

	Reset<forecast_type>();

	while (Next<forecast_type>())
	{
		Reset<forecast_time>();

		while (Next<forecast_time>())
		{
			Reset<level>();

			while (Next<level>())
			{
				Reset<param>();

				while (Next<param>())
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

	otherInfo->Reset<forecast_type>();

	// X = forecast type
	// Y = time
	// Z = level
	// Å = param

	while (otherInfo->Next<forecast_type>())
	{
		if (itsForecastTypeIterator.Add(otherInfo->Value<forecast_type>()))  // no duplicates
		{
			ReIndex(Size<forecast_type>() - 1, Size<forecast_time>(), Size<level>(), Size<param>());
		}

		if (!Find<forecast_type>(otherInfo->Value<forecast_type>()))
		{
			itsLogger.Fatal("Unable to set forecast type, merge failed");
			himan::Abort();
		}

		otherInfo->Reset<forecast_time>();

		while (otherInfo->Next<forecast_time>())
		{
			if (itsTimeIterator.Add(otherInfo->Value<forecast_time>()))  // no duplicates
			{
				ReIndex(Size<forecast_type>(), Size<forecast_time>() - 1, Size<level>(), Size<param>());
			}

			if (!Find<forecast_time>(otherInfo->Value<forecast_time>()))
			{
				itsLogger.Fatal("Unable to set time, merge failed");
				himan::Abort();
			}

			otherInfo->Reset<level>();

			while (otherInfo->Next<level>())
			{
				if (itsLevelIterator.Add(otherInfo->Value<level>()))  // no duplicates
				{
					ReIndex(Size<forecast_type>(), Size<forecast_time>(), Size<level>() - 1, Size<param>());
				}

				if (!Find<level>(otherInfo->Value<level>()))
				{
					itsLogger.Fatal("Unable to set level, merge failed");
					himan::Abort();
				}

				otherInfo->Reset<param>();

				while (otherInfo->Next<param>())
				{
					if (itsParamIterator.Add(otherInfo->Value<param>()))  // no duplicates
					{
						ReIndex(Size<forecast_type>(), Size<forecast_time>(), Size<level>(), Size<param>() - 1);
					}

					if (!Find<param>(otherInfo->Value<param>()))
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

void info::First()
{
	First<level>();
	First<param>();
	First<forecast_time>();
	First<forecast_type>();
	FirstLocation();
}

void info::Reset()
{
	Reset<level>();
	Reset<param>();
	Reset<forecast_time>();
	Reset<forecast_type>();
	ResetLocation();
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
void info::ReIndex(size_t oldForecastTypeSize, size_t oldTimeSize, size_t oldLevelSize, size_t oldParamSize)
{
	vector<shared_ptr<base>> theDimensions(Size<forecast_type>() * Size<forecast_time>() * Size<level>() *
	                                       Size<param>());

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

	if (Next<param>())
	{
		return true;
	}

	// No more params at this forecast type/level/time combination; rewind param iterator

	First<param>();

	if (Next<level>())
	{
		return true;
	}

	// No more levels at this forecast type/time combination; rewind level iterator

	First<level>();

	if (Next<forecast_time>())
	{
		return true;
	}

	// No more times at this forecast type; rewind time iterator, level iterator is
	// already at first place

	First<forecast_time>();

	if (Next<forecast_type>())
	{
		return true;
	}

	return false;
}

bool info::IsValidGrid() const
{
	return (itsDimensions[Index()] != nullptr && Grid());
}

void info::FirstValidGrid()
{
	for (Reset<param>(); Next<param>();)
	{
		if (IsValidGrid())
		{
			return;
		}
	}

	itsLogger.Fatal("A dimension with no valid infos? Madness!");
	himan::Abort();
}

namespace himan
{
template <>
iterator<param>& info::Iterator<param>()
{
	return itsParamIterator;
}
template <>
const iterator<param>& info::Iterator<param>() const
{
	return itsParamIterator;
}
template <>
iterator<level>& info::Iterator<level>()
{
	return itsLevelIterator;
}
template <>
const iterator<level>& info::Iterator<level>() const
{
	return itsLevelIterator;
}
template <>
iterator<forecast_time>& info::Iterator<forecast_time>()
{
	return itsTimeIterator;
}
template <>
const iterator<forecast_time>& info::Iterator<forecast_time>() const
{
	return itsTimeIterator;
}
template <>
iterator<forecast_type>& info::Iterator<forecast_type>()
{
	return itsForecastTypeIterator;
}
template <>
const iterator<forecast_type>& info::Iterator<forecast_type>() const
{
	return itsForecastTypeIterator;
}
}  // namespace himan
