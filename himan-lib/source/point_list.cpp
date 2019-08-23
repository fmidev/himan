/**
 * @file point_list.cpp
 *
 */

#include "point_list.h"
#include "info.h"
#include "logger.h"

using namespace himan;
using namespace std;

point_list::point_list() : irregular_grid(), itsStations()
{
	itsLogger = logger("point_list");
	Type(kPointList);
}

point_list::point_list(const vector<station>& theStations) : irregular_grid(), itsStations(theStations)
{
	itsLogger = logger("point_list");

	Type(kPointList);
}

point_list::point_list(const point_list& other) : irregular_grid(other), itsStations(other.itsStations)
{
	itsLogger = logger("point_list");
}

size_t point_list::Size() const
{
	return itsStations.size();
}
bool point_list::EqualsTo(const point_list& other) const
{
	if (irregular_grid::EqualsTo(other))
	{
		if (itsGridType != other.itsGridType)
		{
			itsLogger.Trace("Projections do not match: " + string(HPGridTypeToString.at(itsGridType)) + " vs " +
			                string(HPGridTypeToString.at(other.itsGridType)));
			return false;
		}

		if (itsStations.size() != other.itsStations.size())
		{
			itsLogger.Trace("Station counts do not match: " + to_string(itsStations.size()) + " vs " +
			                to_string(other.itsStations.size()));
			return false;
		}

		for (size_t i = 0; i < itsStations.size(); i++)
		{
			if (itsStations[i] != other.itsStations[i])
			{
				itsLogger.Trace("Station " + to_string(i) + " does not match: " + static_cast<string>(itsStations[i]) +
				                " vs " + static_cast<string>(other.itsStations[i]));
				return false;
			}
		}
	}

	return true;
}

ostream& point_list::Write(std::ostream& file) const
{
	grid::Write(file);

	for (size_t i = 0; i < itsStations.size(); i++)
	{
		file << "__itsStation__[" << i << "] " << itsStations[i].X() << "," << itsStations[i].Y() << " "
		     << itsStations[i].Id() << endl;
	}

	return file;
}

station point_list::Station(size_t theLocationIndex) const
{
	return itsStations[theLocationIndex];
}
point point_list::LatLon(size_t theLocationIndex) const
{
	return itsStations[theLocationIndex];
}
void point_list::Station(size_t theLocationIndex, const station& theStation)
{
	itsStations[theLocationIndex] = theStation;
}

point point_list::FirstPoint() const
{
	if (itsStations.empty())
	{
		return point();
	}

	return LatLon(0);
}

point point_list::LastPoint() const
{
	if (itsStations.empty())
	{
		return point();
	}

	return LatLon(itsStations.size() - 1);
}

const vector<station>& point_list::Stations() const
{
	return itsStations;
}

void point_list::Stations(const vector<station>& theStations)
{
	itsStations = theStations;
	// itsData.Resize(theStations.size(), 1, 1);
}

size_t point_list::Hash() const
{
	size_t hash = Type();
	for (const auto& station : Stations())
		boost::hash_combine(hash, station.Hash());
	return hash;
}

bool point_list::operator!=(const grid& other) const
{
	return !(other == *this);
}
bool point_list::operator==(const grid& other) const
{
	const point_list* g = dynamic_cast<const point_list*>(&other);

	if (g)
	{
		return EqualsTo(*g);
	}

	return false;
}

unique_ptr<grid> point_list::Clone() const
{
	return unique_ptr<grid>(new point_list(*this));
}
