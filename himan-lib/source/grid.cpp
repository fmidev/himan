/**
 * @file grid.cpp
 *
 * @date Jan 23, 2013
 * @author partio
 */

#include "grid.h"
#include "info.h"
#include <NFmiRotatedLatLonArea.h>
#include <NFmiStereographicArea.h>
#include <NFmiLatLonArea.h>
#include "logger_factory.h"

using namespace himan;
using namespace std;

grid::grid() 
	: itsData(new unpacked())
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
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("grid"));
}

grid::grid(HPScanningMode theScanningMode,
			bool theUVRelativeToGrid,
			HPProjectionType theProjection,
			point theBottomLeft,
			point theTopRight,
			point theSouthPole,
			double theOrientation)
	: itsData(new unpacked())
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
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("grid"));
}

grid::grid(const grid& other)
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

	if (other.itsData)
	{
		itsData = make_shared<unpacked> (*other.itsData);
	}

	if (other.itsPackedData)
	{
		itsPackedData = make_shared<packed_data> (*other.itsPackedData);
	}

	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("grid"));
}

size_t grid::Ni() const
{
	return itsData->SizeX();
}

size_t grid::Nj() const
{
	return itsData->SizeY();
}

void grid::Ni(size_t theNi)
{
	itsData->SizeX(theNi);
}

void grid::Nj(size_t theNj)
{
	itsData->SizeY(theNj);
}

void grid::Di(double theDi)
{
	itsDi = theDi;
}

double grid::Di() const
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

void grid::Dj(double theDj)
{
	itsDj = theDj;
}

double grid::Dj() const
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

shared_ptr<unpacked> grid::Data() const
{
	return itsData;
}

size_t grid::Size() const
{
	return itsData->Size();
}

HPScanningMode grid::ScanningMode() const
{
	return itsScanningMode;
}

void grid::ScanningMode(HPScanningMode theScanningMode)
{
	itsScanningMode = theScanningMode;
}

bool grid::UVRelativeToGrid() const
{
	return itsUVRelativeToGrid;
}

void grid::UVRelativeToGrid(bool theUVRelativeToGrid)
{
	itsUVRelativeToGrid = theUVRelativeToGrid;
}

bool grid::Value(size_t locationIndex, double theValue)
{
	return itsData->Set(locationIndex, theValue) ;
}

double grid::Value(size_t locationIndex) const
{
	return itsData->At(locationIndex);
}

HPProjectionType grid::Projection() const
{
	return itsProjection;
}

void grid::Projection(HPProjectionType theProjection)
{
	itsProjection = theProjection;
}

vector<double> grid::AB() const
{
	return itsAB;
}

void grid::AB(const vector<double>& theAB)
{
	itsAB = theAB;
}

point grid::BottomLeft() const
{
	return itsBottomLeft;
}

point grid::TopRight() const
{
	return itsTopRight;
}

void grid::BottomLeft(const point& theBottomLeft)
{
	itsBottomLeft = theBottomLeft;
}

void grid::TopRight(const point& theTopRight)
{
	itsTopRight = theTopRight;
}

void grid::SouthPole(const point& theSouthPole)
{
	itsSouthPole = theSouthPole;
}

point grid::SouthPole() const
{
	return itsSouthPole;
}

point grid::FirstGridPoint() const
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
		itsLogger->Warning("Calculating first grid point when scanning mode is unknown");
		break;
	}

	return point(x,y);
}

point grid::LastGridPoint() const
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

bool grid::SetCoordinatesFromFirstGridPoint(const point& firstPoint, size_t ni, size_t nj, double di, double dj)
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
		itsLogger->Warning("Calculating first grid point when scanning mode is unknown");
		break;

	}

	return true;
}


double grid::Orientation() const
{
	return itsOrientation;
}

void grid::Orientation(double theOrientation)
{
	itsOrientation = theOrientation;
}

NFmiGrid* grid::ToNewbaseGrid() const
{

	FmiInterpolationMethod interp = kLinearly;
	FmiDirection dir = static_cast<FmiDirection> (itsScanningMode);

	NFmiArea* theArea = 0;

	// Newbase does not understand grib2 longitude coordinates

	double bottomLeftLongitude = itsBottomLeft.X();
	double topRightLongitude = itsTopRight.X();

	if (bottomLeftLongitude > 180 || topRightLongitude > 180)
	{
		bottomLeftLongitude -= 180;
		topRightLongitude -= 180;
	}

	switch (itsProjection)
	{
	case kLatLonProjection:
	{
		theArea = new NFmiLatLonArea(NFmiPoint(bottomLeftLongitude, itsBottomLeft.Y()),
									 NFmiPoint(topRightLongitude, itsTopRight.Y()));

		break;
	}

	case kRotatedLatLonProjection:
	{
		theArea = new NFmiRotatedLatLonArea(NFmiPoint(bottomLeftLongitude, itsBottomLeft.Y()),
											NFmiPoint(topRightLongitude, itsTopRight.Y()),
											NFmiPoint(itsSouthPole.X(), itsSouthPole.Y()),
											NFmiPoint(0.,0.), // default values
											NFmiPoint(1.,1.), // default values
											true);
		break;
	}

	case kStereographicProjection:
	{
		theArea = new NFmiStereographicArea(NFmiPoint(bottomLeftLongitude, itsBottomLeft.Y()),
											NFmiPoint(topRightLongitude, itsTopRight.Y()),
											itsOrientation);
		break;

	}

	default:
		throw runtime_error(ClassName() + ": No supported projection found");
		break;
	}

	assert(theArea);
	
	NFmiGrid* theGrid (new NFmiGrid(theArea, Ni(), Nj(), dir, interp));

	size_t dataSize = itsData->Size();

	if (dataSize)   // Do we have data
	{

		NFmiDataPool thePool;

		float* arr = new float[dataSize];

		// convert double array to float

		const double* src = itsData->ValuesAsPOD();

		copy(src, src + dataSize, arr);

		if (!thePool.Init(dataSize, arr))
		{
			throw runtime_error("NFmiDataPool init failed");
		}

		if (!theGrid->Init(&thePool))
		{
			throw runtime_error("NFmiGrid init failed");
		}

		delete [] arr;
	}

	delete theArea;

	return theGrid;

}

bool grid::operator==(const grid& other) const
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

	return true;

}

bool grid::operator!=(const grid& other) const
{
	return !(*this == other);
}

void grid::Data(shared_ptr<unpacked> d)
{
	itsData = d;
}

bool grid::Stagger(double xStaggerFactor, double yStaggerFactor)
{

	if (xStaggerFactor == 0 && yStaggerFactor == 0)
	{
		return true;
	}

	point bottomLeft, topRight;

	assert(itsProjection != kStereographicProjection);

	bottomLeft.X(itsBottomLeft.X() + xStaggerFactor * Di());
	bottomLeft.Y(itsBottomLeft.Y() + yStaggerFactor * Dj());

	topRight.X(itsTopRight.X() + xStaggerFactor * Di());
	topRight.Y(itsTopRight.Y() + yStaggerFactor * Dj());

	itsBottomLeft = bottomLeft;
	itsTopRight = topRight;

	assert(itsData->Size() > 0);

	//auto newData = unique_ptr<unpacked> (new unpacked(*itsData)); //make_shared<unpacked> (*itsData);
	auto newData = new unpacked(*itsData);

	size_t sizeX = itsData->SizeX(), sizeY = itsData->SizeY();

	const size_t ZI = 1; // Z dimension is always missing here
	
	if (xStaggerFactor == -0.5)
	{

		if (yStaggerFactor == -0.5)
		{
			newData->Set(0, 0, ZI, itsData->At(0, 0));

			for (size_t xi = 1; xi < sizeX; xi++)
			{
				newData->Set(xi, 0, ZI, 0.5 * (itsData->At(xi-1, 0) + itsData->At(xi, 0)));
						
				for (size_t yi = 1; yi < sizeY; yi++)
				{
					newData->Set(0, yi, ZI, 0.5 * (itsData->At(0, yi-1) + itsData->At(0, yi)));

					double val = 0.25 * (	itsData->At(xi-1, yi-1) +
											itsData->At(xi, yi -1) +
											itsData->At(xi-1, yi) +
											itsData->At(xi, yi)	);
					
					newData->Set(xi, yi, ZI, val);
				}
			}

		}
		else if (yStaggerFactor == 0)
		{
			for (size_t yi = 0; yi < sizeY; yi++)
			{
				newData->Set(0, yi, ZI, itsData->At(0, yi));
				newData->Set(0, yi, ZI, 0.5 * (3 * itsData->At(0, yi) - itsData->At(1, yi))); // Lefternmost column is extrapolated

				for (size_t xi = 1; xi < sizeX; xi++)
				{
					newData->Set(xi, yi, ZI, 0.5 * (itsData->At(xi-1, yi) + itsData->At(xi, yi)));
				}
			}

		}
		else if (yStaggerFactor == 0.5)
		{
			newData->Set(0, sizeY-1, ZI, itsData->At(0, sizeY-1));

			for (size_t xi = 1; xi < sizeX; xi++)
			{
				newData->Set(xi, sizeY-1, ZI, 0.5 * (itsData->At(xi-1, sizeY-1) + itsData->At(xi, sizeY-1)));

				for (size_t yi = 0; yi < sizeY-1; yi++)
				{
					newData->Set(0, yi, ZI, 0.5 * (itsData->At(0, yi) + itsData->At(0, yi+1)));

					double val = 0.25 * (	itsData->At(xi-1, yi) +
											itsData->At(xi, yi) +
											itsData->At(xi-1, yi+1) +
											itsData->At(xi, yi+1)	);

					newData->Set(xi, yi, ZI, val);
				}
			}
		}
		else
		{
			throw runtime_error(ClassName() + ": Invalid y stagger value: " + boost::lexical_cast<string> (yStaggerFactor));
		}
	}
	else if (xStaggerFactor == 0)
	{
		if (yStaggerFactor == -0.5)
		{
			for (size_t xi = 0; xi < sizeX; xi++)
			{
				newData->Set(xi, 0, ZI, itsData->At(xi, 0));

				for (size_t yi = 1; yi < sizeY; yi++)
				{
					newData->Set(xi, yi, ZI, 0.5 * (itsData->At(xi, yi-1) + itsData->At(xi, yi)));
				}
			}
		}
		else if (yStaggerFactor == 0.5)
		{
			for (size_t xi = 0; xi < sizeX; xi++)
			{
				newData->Set(xi, sizeY-1, ZI, itsData->At(xi, sizeY-1));

				for (size_t yi = 0; yi < sizeY-1; yi++)
				{
					newData->Set(xi, yi, ZI, 0.5 * (itsData->At(xi, yi) + itsData->At(xi, yi+1)));
				}
			}
		}
		else
		{
			throw runtime_error(ClassName() + ": Invalid y stagger value: " + boost::lexical_cast<string> (yStaggerFactor));
		}
	}
	else if (xStaggerFactor == 0.5)
	{
		if (yStaggerFactor == -0.5)
		{
			newData->Set(sizeX-1, 0, ZI, itsData->At(sizeX-1, 0));

			for (size_t xi = 0; xi < sizeX-1; xi++)
			{
				newData->Set(xi, 0, ZI, 0.5 * (itsData->At(xi,0) + itsData->At(xi+1, 0)));

				for (size_t yi = 1; yi < sizeY; yi++)
				{
					newData->Set(sizeX-1, yi, ZI, 0.5 * (itsData->At(sizeX-1, yi-1) + itsData->At(sizeX-1, yi)));

					double val = 0.25 * (	itsData->At(xi, yi-1) +
											itsData->At(xi+1, yi-1) +
											itsData->At(xi, yi) +
											itsData->At(xi+1, yi)	);

					newData->Set(xi, yi, ZI, val);
				}
			}
		}

		if (yStaggerFactor == 0)
		{
			for (size_t yi = 0; yi < sizeY; yi++)
			{
				newData->Set(sizeX-1, yi, ZI, 0.5 * (3 * itsData->At(sizeX-1, yi) - itsData->At(sizeX-2, yi))); // Righternmost column is extrapolated
		
				for (size_t xi = 0; xi < sizeX-1; xi++)
				{
					newData->Set(xi, yi, ZI, 0.5 * (itsData->At(xi, yi) + itsData->At(xi+1, yi)));
				}
			}
		}
		else if (yStaggerFactor == 0.5)
		{
			newData->Set(sizeX-1, sizeY-1, ZI, itsData->At(sizeX-1, sizeY-1));

			for (size_t xi = 0; xi < sizeX-1; xi++)
			{
				newData->Set(xi, sizeY-1, ZI, 0.5 * (itsData->At(xi,sizeY-1) + itsData->At(xi+1, sizeY-1)));

				for (size_t yi = 0; yi < sizeY-1; yi++)
				{
					newData->Set(sizeX-1, yi, ZI, 0.5 * (itsData->At(sizeX-1, yi) + itsData->At(sizeX-1, yi+1)));

					double val = 0.25 * (	itsData->At(xi, yi) +
											itsData->At(xi+1, yi) +
											itsData->At(xi, yi+1) +
											itsData->At(xi+1, yi+1)	);

					newData->Set(xi, yi, ZI, val);
				}
			}
		}
		else
		{
			throw runtime_error(ClassName() + ": Invalid y stagger value: " + boost::lexical_cast<string> (yStaggerFactor));
		}
	}
	else
	{
		throw runtime_error(ClassName() + ": Invalid x stagger value: " + boost::lexical_cast<string> (xStaggerFactor));
	}

	itsData = unique_ptr<unpacked> (newData);

	return true;
}

bool grid::Swap(HPScanningMode newScanningMode)
{
	if (itsScanningMode == newScanningMode)
	{
		itsLogger->Trace("Not swapping data between same scanningmodes");
		return true;
	}

	assert(itsData);

	// Flip with regards to x axis

	if ((itsScanningMode == kTopLeft && newScanningMode == kBottomLeft) || (itsScanningMode == kBottomLeft && newScanningMode == kTopLeft))
	{
		size_t halfSize = static_cast<size_t> (floor(Nj()/2));
		
		for (size_t y = 0; y < halfSize; y++)
		{
			for (size_t x = 0; x < Ni(); x++)
			{
				double upper = itsData->At(x,y);
				double lower = itsData->At(x, Nj()-1-y);

				itsData->Set(x,y,0,lower);
				itsData->Set(x,Nj()-1-y,0,upper);
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

shared_ptr<packed_data> grid::PackedData() const
{
	return itsPackedData;
}

void grid::PackedData(shared_ptr<packed_data> thePackedData)
{
	itsPackedData = thePackedData;
}

bool grid::IsPackedData() const
{
	if (itsPackedData && itsPackedData->HasData())
	{
		return true;
	}
	
	return false;
}

ostream& grid::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;

	if (itsData)
	{
		file << *itsData;
	}
	
	file << "__dataIsPacked__ " << IsPackedData() << endl;
	file << "__itsScanningMode__ " << itsScanningMode << endl;
	file << "__itsUVRelativeToGrid__ " << itsUVRelativeToGrid << endl;
	file << "__itsProjection__ " << itsProjection << endl;

	for (size_t i = 0; i < itsAB.size(); i++)
	{
		cout << "__itsAB__" << itsAB[i] << endl;
	}

	file << itsBottomLeft;
	file << itsTopRight;
	file << itsSouthPole;
	file << "__itsOrientation__ " << itsOrientation << endl;
	file << "__itsDi__ " << itsDi << endl;
	file << "__itsDj__ " << itsDj << endl;


	return file;
}
