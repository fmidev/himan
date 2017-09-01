/**
 * @file point_list.cpp
 *
 */

#include "point_list.h"
#include "info.h"
#include "logger.h"
#include <NFmiGrid.h>
#include <NFmiLatLonArea.h>
#include <NFmiRotatedLatLonArea.h>
#include <NFmiStereographicArea.h>

using namespace himan;
using namespace std;

point_list::point_list() : grid(kIrregularGrid, kPointList), itsStations()
{
	itsLogger = logger("point_list");
}

point_list::point_list(const vector<station>& theStations) : grid(kIrregularGrid, kPointList), itsStations(theStations)
{
	itsLogger = logger("point_list");

	itsData.Resize(theStations.size(), 1, 1);

	assert(itsData.Size() == theStations.size());
}

point_list::point_list(const point_list& other) : grid(other), itsStations(other.itsStations)
{
	itsLogger = logger("point_list");
}

size_t point_list::Size() const { return itsData.Size(); }
bool point_list::EqualsTo(const point_list& other) const
{
	if (grid::EqualsTo(other))
	{
		if (itsGridType != other.itsGridType)
		{
			itsLogger.Trace("Projections do not match: " + string(HPGridTypeToString.at(itsGridType)) + " vs " +
			                 string(HPGridTypeToString.at(other.itsGridType)));
			return false;
		}

		if (itsStations.size() != other.itsStations.size())
		{
			itsLogger.Trace("Station counts do not match: " + to_string(itsStations.size()) +
			                 " vs " + to_string(other.itsStations.size()));
			return false;
		}

		for (size_t i = 0; i < itsStations.size(); i++)
		{
			if (itsStations[i] != other.itsStations[i])
			{
				itsLogger.Trace("Station " + to_string(i) + " does not match");
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
		file << "__itsPoint__[" << i << "] " << itsStations[i].X() << "," << itsStations[i].Y() << endl;
	}

	return file;
}

station point_list::Station(size_t theLocationIndex) const { return itsStations[theLocationIndex]; }
point point_list::LatLon(size_t theLocationIndex) const { return itsStations[theLocationIndex]; }
void point_list::Station(size_t theLocationIndex, const station& theStation)
{
	itsStations[theLocationIndex] = theStation;
}

const vector<station>& point_list::Stations() const { return itsStations; }
void point_list::Stations(const vector<station>& theStations)
{
	itsStations = theStations;
	itsData.Resize(theStations.size(), 1, 1);
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

point point_list::BottomLeft() const
{
	// Maybe a useless implementation, but nevertheless: return the most western and
	// southern point

	point ret;

	for (const auto& p : itsStations)
	{
		if (IsKHPMissingValue(ret.X()) || p.X() < ret.X())
		{
			if (IsKHPMissingValue(ret.Y()) || p.Y() > ret.Y())
			{
				ret = p;
			}
		}
	}

	return ret;
}

point point_list::TopRight() const
{
	point ret;

	for (const auto& p : itsStations)
	{
		if (IsKHPMissingValue(ret.X()) || p.X() > ret.X())
		{
			if (IsKHPMissingValue(ret.Y()) || p.Y() < ret.Y())
			{
				ret = p;
			}
		}
	}

	return ret;
}

HPScanningMode point_list::ScanningMode() const { return kUnknownScanningMode; }
bool point_list::Swap(HPScanningMode newScanningMode)
{
	// can't be done here
	return false;
}

double point_list::Di() const { return kHPMissingValue; }
double point_list::Dj() const { return kHPMissingValue; }
size_t point_list::Ni() const { return itsData.Size(); }
size_t point_list::Nj() const { return 1; }
bool point_list::operator!=(const point_list& other) const { return !(other == *this); }
bool point_list::operator==(const point_list& other) const
{
	const point_list* g = dynamic_cast<const point_list*>(&other);

	if (g)
	{
		return EqualsTo(*g);
	}

	return false;
}

point_list* point_list::Clone() const { return new point_list(*this); }
point point_list::XY(const point& latlon) const
{
	// Linear search
	// Returned value is in 1D vector format, ie. Y = 0.
	int x = 0;
	for (const auto& p : itsStations)
	{
		if (p == station(p.Id(), p.Name(), latlon.X(), latlon.Y()))
		{
			return point(x, 0);
		}

		x++;
	}

	return point();
}
