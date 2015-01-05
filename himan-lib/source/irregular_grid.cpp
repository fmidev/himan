/**
 * @file irregular_grid.cpp
 *
 * @date Jan 23, 2013
 * @author partio
 */

#include "irregular_grid.h"
#include "info.h"
#include <NFmiRotatedLatLonArea.h>
#include <NFmiStereographicArea.h>
#include <NFmiLatLonArea.h>
#include "logger_factory.h"
#include <NFmiGrid.h>

using namespace himan;
using namespace std;

irregular_grid::irregular_grid()
	: itsData()
	, itsScanningMode(kUnknownScanningMode)
	, itsUVRelativeToGrid(false)
	, itsProjection(kUnknownProjection)
	, itsAB()
	, itsBottomLeft()
	, itsTopRight()
	, itsSouthPole()
	, itsOrientation(kHPMissingValue)
{
	itsGridType = kIrregularGrid;
	itsLogger = logger_factory::Instance()->GetLog("irregular_grid");
}

irregular_grid::irregular_grid(const vector<station>& theStations)
	: itsData()
	, itsStations(theStations)
{
	itsData.Resize(theStations.size(), 1, 1);
	
	itsGridType = kIrregularGrid;

	assert(itsData.Size() == theStations.size());
	
	itsLogger = logger_factory::Instance()->GetLog("irregular_grid");
}

irregular_grid::irregular_grid(const irregular_grid& other)
{
	itsUVRelativeToGrid = other.itsUVRelativeToGrid;
	itsProjection = other.itsProjection;
	itsAB = other.itsAB;
	itsSouthPole = other.itsSouthPole;
	itsOrientation = other.itsOrientation;
	itsData = other.itsData;
	itsGridType = other.itsGridType;
	itsStations = other.itsStations;
	
	itsLogger = logger_factory::Instance()->GetLog("irregular_grid");
}

unpacked& irregular_grid::Data()
{
	return itsData;
}

size_t irregular_grid::Size() const
{
	return itsData.Size();
}

bool irregular_grid::UVRelativeToGrid() const
{
	return itsUVRelativeToGrid;
}

void irregular_grid::UVRelativeToGrid(bool theUVRelativeToGrid)
{
	itsUVRelativeToGrid = theUVRelativeToGrid;
}

bool irregular_grid::Value(size_t locationIndex, double theValue)
{
	return itsData.Set(locationIndex, theValue) ;
}

double irregular_grid::Value(size_t locationIndex) const
{
	return double(itsData.At(locationIndex));
}

HPProjectionType irregular_grid::Projection() const
{
	return itsProjection;
}

void irregular_grid::Projection(HPProjectionType theProjection)
{
	itsProjection = theProjection;
}

vector<double> irregular_grid::AB() const
{
	return itsAB;
}

void irregular_grid::AB(const vector<double>& theAB)
{
	itsAB = theAB;
}

point irregular_grid::BottomLeft() const
{
	return itsBottomLeft;
}

point irregular_grid::TopRight() const
{
	return itsTopRight;
}

void irregular_grid::SouthPole(const point& theSouthPole)
{
	itsSouthPole = theSouthPole;
}

point irregular_grid::SouthPole() const
{
	return itsSouthPole;
}

double irregular_grid::Orientation() const
{
	return itsOrientation;
}

void irregular_grid::Orientation(double theOrientation)
{
	itsOrientation = theOrientation;
}

bool irregular_grid::EqualsTo(const irregular_grid& other) const
{
	if (grid::EqualsTo(other))
	{
		if (itsProjection != other.itsProjection)
		{
			itsLogger->Trace("Projections do not match: " + string(HPProjectionTypeToString.at(itsProjection)) + " vs " + string(HPProjectionTypeToString.at(other.itsProjection)));
			return false;
		}

		if (itsProjection == kRotatedLatLonProjection)
		{
			if (itsSouthPole != other.SouthPole())
			{
				itsLogger->Trace("SouthPole does not match: X " + boost::lexical_cast<string> (itsSouthPole.X()) + " vs " + boost::lexical_cast<string> (other.SouthPole().X()));
				itsLogger->Trace("SouthPole does not match: Y " + boost::lexical_cast<string> (itsSouthPole.Y()) + " vs " + boost::lexical_cast<string> (other.SouthPole().Y()));
				return false;
			}
		}

		if (itsProjection == kStereographicProjection)
		{
			if (itsOrientation != other.Orientation())
			{
				itsLogger->Trace("Orientations don't match: " + boost::lexical_cast<string> (itsOrientation) + " vs " + boost::lexical_cast<string> (other.Orientation()));
				return false;
			}
		}
		
		if (itsStations.size() != other.itsStations.size())
		{
			itsLogger->Trace("Station counts do not match: " + boost::lexical_cast<string> (itsStations.size()) + " vs " + boost::lexical_cast<string> (other.itsStations.size()));
			return false;
		}
		
		for (size_t i = 0; i < itsStations.size(); i++)
		{
			if (itsStations[i] != other.itsStations[i])
			{
				itsLogger->Trace("Station " + boost::lexical_cast<string> (i) + " does not match");
				return false;
			}
		}
	}
	
	return true;	
}
		
void irregular_grid::Data(const unpacked& d)
{
	itsData = d;
}

ostream& irregular_grid::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;

	file << itsData;
	
	file << "__itsScanningMode__ " << HPScanningModeToString.at(itsScanningMode) << endl;
	file << "__itsUVRelativeToGrid__ " << itsUVRelativeToGrid << endl;
	file << "__itsProjection__ " << HPProjectionTypeToString.at(itsProjection) << endl;

	for (size_t i = 0; i < itsAB.size(); i++)
	{
		file << "__itsAB__" << itsAB[i] << endl;
	}

	file << itsBottomLeft;
	file << itsTopRight;
	file << itsSouthPole;
	file << "__itsOrientation__ " << itsOrientation << endl;
	
	for (size_t i = 0; i < itsStations.size(); i++)
	{
		file << "__itsPoint__[" << i << "] " << itsStations[i].X() << "," << itsStations[i].Y() << endl;
	}

	return file;
}

station irregular_grid::Station(size_t locationIndex) const
{
	return itsStations[locationIndex];
}

point irregular_grid::LatLon(size_t locationIndex) const
{
	return itsStations[locationIndex];
}

void irregular_grid::Station(size_t locationIndex, const station& theStation)
{
	itsStations[locationIndex] = theStation;
}

const vector<station>& irregular_grid::Stations() const
{
	return itsStations;
}

void irregular_grid::Stations(const vector<station>& theStations)
{
	itsStations = theStations;
}

bool irregular_grid::operator!=(const grid& other) const
{
	return !(other == *this);
}

bool irregular_grid::operator==(const grid& other) const
{
	const irregular_grid* g = dynamic_cast<const irregular_grid*> (&other);
	
	if (g)
	{
		return EqualsTo(*g);
	}
	
	return false;
}
