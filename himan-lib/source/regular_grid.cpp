/**
 * @file regular_grid.cpp
 *
 * @date Jan 23, 2013
 * @author partio
 */

#include "regular_grid.h"
#include "info.h"
#include <NFmiRotatedLatLonArea.h>
#include <NFmiStereographicArea.h>
#include <NFmiLatLonArea.h>
#include "logger_factory.h"

using namespace himan;
using namespace std;

regular_grid::regular_grid()
	: itsData()
	, itsPackedData()
	, itsScanningMode(kUnknownScanningMode)
	, itsUVRelativeToGrid(false)
	, itsProjection(kUnknownProjection)
	, itsAB()
	, itsBottomLeft()
	, itsTopRight()
	, itsSouthPole()
	, itsOrientation(kHPMissingValue)
	, itsDi(kHPMissingValue)
	, itsDj(kHPMissingValue)
{
	itsGridType = kRegularGrid;
	itsLogger = logger_factory::Instance()->GetLog("regular_grid");
}

regular_grid::regular_grid(HPScanningMode theScanningMode,
			bool theUVRelativeToGrid,
			HPProjectionType theProjection,
			point theBottomLeft,
			point theTopRight,
			point theSouthPole,
			double theOrientation)
	: itsData()
	, itsPackedData()
	, itsScanningMode(theScanningMode)
	, itsUVRelativeToGrid(theUVRelativeToGrid)
	, itsProjection(theProjection)
	, itsAB()
	, itsBottomLeft(theBottomLeft)
	, itsTopRight(theTopRight)
	, itsSouthPole(theSouthPole)
	, itsOrientation(theOrientation)
	, itsDi(kHPMissingValue)
	, itsDj(kHPMissingValue)
{
	itsGridType = kRegularGrid;
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("regular_grid"));
}

regular_grid::regular_grid(const regular_grid& other)
{
	itsScanningMode = other.itsScanningMode;
	itsUVRelativeToGrid = other.itsUVRelativeToGrid;
	itsProjection = other.itsProjection;
	itsAB = other.itsAB;
	itsBottomLeft = other.itsBottomLeft;
	itsTopRight = other.itsTopRight;
	itsSouthPole = other.itsSouthPole;
	itsOrientation = other.itsOrientation;
	itsDi = other.itsDi;
	itsDj = other.itsDj;
	itsData = other.itsData;
	itsGridType = other.itsGridType;
	
#ifdef HAVE_CUDA

	if (other.itsPackedData)
	{
		switch (other.itsPackedData->packingType)
		{
		case kSimplePacking:
			itsPackedData = unique_ptr<simple_packed> (new simple_packed(*dynamic_cast<simple_packed*> (other.itsPackedData.get())));
			break;

		default:
			itsPackedData = unique_ptr<packed_data> (new packed_data(*itsPackedData));
			break;
		}
		
	}

#endif
	
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("regular_grid"));
}

size_t regular_grid::Ni() const
{
	return itsData.SizeX();
}

size_t regular_grid::Nj() const
{
	return itsData.SizeY();
}

void regular_grid::Ni(size_t theNi)
{
	itsData.SizeX(theNi);
}

void regular_grid::Nj(size_t theNj)
{
	itsData.SizeY(theNj);
}

void regular_grid::Di(double theDi)
{
	itsDi = theDi;
}

double regular_grid::Di() const
{
	if (itsDi == kHPMissingValue)
	{
		assert(itsProjection != kStereographicProjection);
		
		assert(itsBottomLeft.X() != static_cast<size_t> (kHPMissingInt));
		assert(itsTopRight.X() != static_cast<size_t> (kHPMissingInt));
		itsDi = fabs((itsBottomLeft.X() - itsTopRight.X()) / (static_cast<double> (Ni())-1));
	}

	return itsDi;
}

void regular_grid::Dj(double theDj)
{
	itsDj = theDj;
}

double regular_grid::Dj() const
{
	if (itsDj == kHPMissingValue)
	{
		assert(itsProjection != kStereographicProjection);

		assert(itsBottomLeft.Y() != kHPMissingInt);
		assert(itsTopRight.Y() != kHPMissingInt);
		itsDj = fabs((itsBottomLeft.Y() - itsTopRight.Y()) / (static_cast<double> (Nj())-1));
	}

	return itsDj;
}

unpacked& regular_grid::Data()
{
	return itsData;
}

size_t regular_grid::Size() const
{
	return itsData.Size();
}

HPScanningMode regular_grid::ScanningMode() const
{
	return itsScanningMode;
}

void regular_grid::ScanningMode(HPScanningMode theScanningMode)
{
	itsScanningMode = theScanningMode;
}

bool regular_grid::UVRelativeToGrid() const
{
	return itsUVRelativeToGrid;
}

void regular_grid::UVRelativeToGrid(bool theUVRelativeToGrid)
{
	itsUVRelativeToGrid = theUVRelativeToGrid;
}

bool regular_grid::Value(size_t locationIndex, double theValue)
{
	return itsData.Set(locationIndex, theValue) ;
}

double regular_grid::Value(size_t locationIndex) const
{
	return double(itsData.At(locationIndex));
}

HPProjectionType regular_grid::Projection() const
{
	return itsProjection;
}

void regular_grid::Projection(HPProjectionType theProjection)
{
	itsProjection = theProjection;
}

vector<double> regular_grid::AB() const
{
	return itsAB;
}

void regular_grid::AB(const vector<double>& theAB)
{
	itsAB = theAB;
}

point regular_grid::BottomLeft() const
{
	return itsBottomLeft;
}

point regular_grid::TopRight() const
{
	return itsTopRight;
}

void regular_grid::BottomLeft(const point& theBottomLeft)
{
	itsBottomLeft = theBottomLeft;
}

void regular_grid::TopRight(const point& theTopRight)
{
	itsTopRight = theTopRight;
}

void regular_grid::SouthPole(const point& theSouthPole)
{
	itsSouthPole = theSouthPole;
}

point regular_grid::SouthPole() const
{
	return itsSouthPole;
}

point regular_grid::FirstGridPoint() const
{
	double x = kHPMissingValue;
	double y = kHPMissingValue;

	if (itsProjection == kStereographicProjection)
	{
		// Currently support no other scanning mode than bottom left for stereographic projections
		assert(itsScanningMode == kBottomLeft);
		return itsBottomLeft;
	}

	assert(itsBottomLeft.X() != kHPMissingValue);
	assert(itsBottomLeft.Y() != kHPMissingValue);
	assert(itsTopRight.X() != kHPMissingValue);
	assert(itsTopRight.Y() != kHPMissingValue);
	assert(Ni() > 0);
	assert(Nj() > 0);

	switch (itsScanningMode)
	{
	case kBottomLeft:
		x = itsBottomLeft.X();
		y = itsBottomLeft.Y();
		break;

	case kTopLeft:
		x = itsTopRight.X() - (static_cast<double> (Ni())-1)*Di();
		y = itsBottomLeft.Y() + (static_cast<double> (Nj())-1)*Dj();
		break;

	case kTopRight:
		x = itsTopRight.X();
		y = itsTopRight.Y();
		break;

	case kBottomRight:
		x = itsBottomLeft.X() + (static_cast<double> (Ni())-1)*Di();
		y = itsTopRight.Y() - (static_cast<double> (Nj())-1)*Dj();
		break;

	default:
		itsLogger->Warning("Calculating first regular_grid point when scanning mode is unknown");
		break;
	}

	return point(x,y);
}

point regular_grid::LastGridPoint() const
{
	if (itsProjection == kStereographicProjection)
	{
		// Currently support no other scanning mode than bottom left for stereographic projections
		assert(itsScanningMode == kBottomLeft);
		return itsTopRight;
	}

	point firstGridPoint = FirstGridPoint();
	point lastGridPoint;

	switch (itsScanningMode)
	{
		case kBottomLeft:
			lastGridPoint.X(firstGridPoint.X() + (static_cast<double> (Ni())-1)*Di());
			lastGridPoint.Y(firstGridPoint.Y() + (static_cast<double> (Nj())-1)*Dj());
			break;

		case kTopLeft:
			lastGridPoint.X(firstGridPoint.X() + (static_cast<double> (Ni())-1)*Di());
			lastGridPoint.Y(firstGridPoint.Y() - (static_cast<double> (Nj())-1)*Dj());
			break;

		default:
			throw runtime_error(ClassName() + ": Invalid scanning mode in LastGridPoint()");
			break;
	}
	
	return lastGridPoint;
}

bool regular_grid::SetCoordinatesFromFirstGridPoint(const point& firstPoint, size_t ni, size_t nj, double di, double dj)
{
	assert(itsScanningMode != kUnknownScanningMode);

	double dni = static_cast<double> (ni) - 1;
	double dnj = static_cast<double> (nj) - 1;

	point topLeft, bottomRight;

	switch (itsScanningMode)
	{
	case kBottomLeft:
		itsBottomLeft = firstPoint;
		itsTopRight = point(itsBottomLeft.X() + dni*di, itsBottomLeft.Y() + dnj*dj);
		break;

	case kTopLeft: // +x-y
		bottomRight = point(firstPoint.X() + dni*di, firstPoint.Y() - dnj*dj);
		itsBottomLeft = point(bottomRight.X() - dni*di, firstPoint.Y() - dnj*dj);
		itsTopRight = point(itsBottomLeft.X() + dni*di, itsBottomLeft.Y() + dnj*dj);
		break;

	case kTopRight: // -x-y
		itsTopRight = firstPoint;
		itsBottomLeft = point(itsTopRight.X() - dni*di, itsTopRight.Y() - dnj*dj);
		break;

	case kBottomRight: // -x+y
		topLeft = point(firstPoint.X() - dni*di, firstPoint.Y() + dnj*dj);
		itsBottomLeft = point(firstPoint.X() - dni*di, topLeft.Y() - dnj*dj);
		itsTopRight = point(itsBottomLeft.X() + dni*di, itsBottomLeft.Y() + dnj*dj);
		break;

	default:
		itsLogger->Warning("Calculating first regular_grid point when scanning mode is unknown");
		break;

	}

	return true;
}


double regular_grid::Orientation() const
{
	return itsOrientation;
}

void regular_grid::Orientation(double theOrientation)
{
	itsOrientation = theOrientation;
}

void regular_grid::Data(const unpacked& d)
{
	itsData = d;
}

bool regular_grid::Swap(HPScanningMode newScanningMode)
{
	if (itsScanningMode == newScanningMode)
	{
		itsLogger->Trace("Not swapping data between same scanningmodes");
		return true;
	}

	assert(itsData.Size());

	// Flip with regards to x axis

	if ((itsScanningMode == kTopLeft && newScanningMode == kBottomLeft) || (itsScanningMode == kBottomLeft && newScanningMode == kTopLeft))
	{
		size_t halfSize = static_cast<size_t> (floor(Nj()/2));
		
		for (size_t y = 0; y < halfSize; y++)
		{
			for (size_t x = 0; x < Ni(); x++)
			{
				double upper = itsData.At(x,y);
				double lower = itsData.At(x, Nj()-1-y);

				itsData.Set(x,y,0,lower);
				itsData.Set(x,Nj()-1-y,0,upper);
			}
		}
	}
	else
	{
		itsLogger->Error("Swap from mode " + string(HPScanningModeToString.at(itsScanningMode)) + " to mode " + string(HPScanningModeToString.at(newScanningMode)) + " not implemented yet");
		return false;
	}

	itsScanningMode = newScanningMode;

	return true;

}

packed_data& regular_grid::PackedData()
{
	assert(itsPackedData);
	return *itsPackedData;
}

void regular_grid::PackedData(unique_ptr<packed_data> thePackedData)
{
	itsPackedData = move(thePackedData);
}

bool regular_grid::IsPackedData() const
{
	if (itsPackedData && itsPackedData->HasData())
	{
		return true;
	}
	
	return false;
}

ostream& regular_grid::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;

	file << itsData;
	
	file << "__dataIsPacked__ " << IsPackedData() << endl;
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
	file << "__itsDi__ " << itsDi << endl;
	file << "__itsDj__ " << itsDj << endl;


	return file;
}

point regular_grid::LatLon(size_t locationIndex) const
{
	assert(Projection() == kLatLonProjection || Projection() == kRotatedLatLonProjection);

	double j;
	point firstPoint = FirstGridPoint();

	if (ScanningMode() == kBottomLeft) //opts.j_scans_positive)
	{
		j = floor(static_cast<double> (locationIndex / itsData.SizeX()));
	}
	else if (ScanningMode() == kTopLeft)
	{
		j = static_cast<double> (Nj()) - floor(static_cast<double> (locationIndex / itsData.SizeX()));
	}
	else
	{
		throw runtime_error("Unsupported scanning mode: " + string(HPScanningModeToString.at(ScanningMode())));
	}

	double i = static_cast<double> (locationIndex) - j * static_cast<double> (Ni());

	return point(firstPoint.X() + i * Di(), firstPoint.Y() + j * Dj());
}


bool regular_grid::operator!=(const grid& other) const
{
	return !(other == *this);
}

bool regular_grid::operator==(const grid& other) const
{
	const regular_grid* g = dynamic_cast<const regular_grid*> (&other);
	
	if (g)
	{
		return EqualsTo(*g);
	}
	
	return false;
}

bool regular_grid::EqualsTo(const regular_grid& other) const
{
	if (grid::EqualsTo(other))
	{
		if (itsProjection != other.itsProjection)
		{
			itsLogger->Trace("Projections do not match: " + string(HPProjectionTypeToString.at(itsProjection)) + " vs " + string(HPProjectionTypeToString.at(other.itsProjection)));
			return false;
		}

		if (itsBottomLeft != other.BottomLeft())
		{
			itsLogger->Trace("BottomLeft does not match: X " + boost::lexical_cast<string> (itsBottomLeft.X()) + " vs " + boost::lexical_cast<string> (other.BottomLeft().X()));
			itsLogger->Trace("BottomLeft does not match: Y " + boost::lexical_cast<string> (itsBottomLeft.Y()) + " vs " + boost::lexical_cast<string> (other.BottomLeft().Y()));
			return false;
		}

		if (itsTopRight != other.TopRight())
		{
			itsLogger->Trace("TopRight does not match: X " + boost::lexical_cast<string> (itsTopRight.X()) + " vs " + boost::lexical_cast<string> (other.TopRight().X()));
			itsLogger->Trace("TopRight does not match: Y " + boost::lexical_cast<string> (itsTopRight.Y()) + " vs " + boost::lexical_cast<string> (other.TopRight().Y()));
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

		if (Ni() != other.Ni())
		{
			itsLogger->Trace("Grid X-counts don't match: " + boost::lexical_cast<string> (Ni()) + " vs " + boost::lexical_cast<string> (other.Ni()));
			return false;
		}

		if (Nj() != other.Nj())
		{
			itsLogger->Trace("Grid Y-counts don't match: " + boost::lexical_cast<string> (Nj()) + " vs " + boost::lexical_cast<string> (other.Nj()));
			return false;
		}

	}
	
	return true;	
}