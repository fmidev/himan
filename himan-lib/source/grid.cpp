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

grid::grid() : itsData(new d_matrix_t(0,0)), itsScanningMode(kUnknownScanningMode), itsUVRelativeToGrid(false)
{
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("grid"));
}

grid::grid(HPScanningMode theScanningMode,
			bool theUVRelativeToGrid,
			HPProjectionType theProjection,
			point theBottomLeft,
			point theTopRight,
			point theSouthPole,
			double theOrientation)
	: itsData(new d_matrix_t(0,0))
	, itsScanningMode(theScanningMode)
	, itsUVRelativeToGrid(theUVRelativeToGrid)
	, itsProjection(theProjection)
	, itsBottomLeft(theBottomLeft)
	, itsTopRight(theTopRight)
	, itsSouthPole(theSouthPole)
	, itsOrientation(theOrientation)
	, itsDi(kHPMissingFloat)
	, itsDj(kHPMissingFloat)
{
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("grid"));
}

grid::grid(const grid& other)
	: itsData(new d_matrix_t(other.Ni(), other.Nj()))
{
	itsScanningMode = other.itsScanningMode;
	itsUVRelativeToGrid = other.itsUVRelativeToGrid;
	itsProjection = other.itsProjection;
	itsBottomLeft = other.itsBottomLeft;
	itsTopRight = other.itsTopRight;
	itsSouthPole = other.itsSouthPole;
	itsOrientation = other.itsOrientation;
	itsDi = other.itsDi;
	itsDj = other.itsDj;

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("grid"));
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
	if (itsDi == kHPMissingFloat)
	{
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
	if (itsDj == kHPMissingFloat)
	{
		assert(itsBottomLeft.Y() != kHPMissingInt);
		assert(itsTopRight.Y() != kHPMissingInt);
		itsDj = fabs((itsBottomLeft.Y() - itsTopRight.Y()) / (static_cast<double> (Nj())-1));
	}

	return itsDj;
}

std::shared_ptr<d_matrix_t> grid::Data() const
{
	return itsData;
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
	double x = kHPMissingFloat;
	double y = kHPMissingFloat;

	if (itsProjection != kLatLonProjection && itsProjection != kRotatedLatLonProjection)
	{
		itsLogger->Warning("Calculating latitude for first gridpoint in non-latlon projection not supported");
		return itsBottomLeft;
	}

	assert(itsBottomLeft.X() != kHPMissingFloat);
	assert(itsBottomLeft.Y() != kHPMissingFloat);
	assert(itsTopRight.X() != kHPMissingFloat);
	assert(itsTopRight.Y() != kHPMissingFloat);
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
	point firstGridPoint = FirstGridPoint();
	return point(firstGridPoint.X() + (static_cast<double> (Ni())-1)*Di(), firstGridPoint.Y() + (static_cast<double> (Nj())-1)*Dj());
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
        throw std::runtime_error(ClassName() + ": No supported projection found");
        break;
    }

    NFmiGrid* theGrid (new NFmiGrid(theArea, Ni(), Nj(), dir, interp));

    size_t dataSize = itsData->Size();

    if (dataSize)   // Do we have data
    {

        NFmiDataPool thePool;

        float* arr = new float[dataSize];

        // convert double array to float

        for (unsigned int i = 0; i < dataSize; i++)
        {
            arr[i] = static_cast<float> (itsData->At(i));
        }

        if (!thePool.Init(dataSize, arr))
        {
            throw std::runtime_error("DataPool init failed");
        }

        if (!theGrid->Init(&thePool))
        {
            throw std::runtime_error("Grid data init failed");
        }

        delete [] arr;
    }

    delete theArea;

    return theGrid;

}

bool grid::operator==(const grid& other) const
{

    if (itsBottomLeft != other.BottomLeft())
    {
        itsLogger->Trace("BottomLeft does not match: " + boost::lexical_cast<std::string> (itsBottomLeft.X()) + " vs " + boost::lexical_cast<std::string> (other.BottomLeft().X()));
        itsLogger->Trace("BottomLeft does not match: " + boost::lexical_cast<std::string> (itsBottomLeft.Y()) + " vs " + boost::lexical_cast<std::string> (other.BottomLeft().Y()));
        return false;
    }

    if (itsTopRight != other.TopRight())
    {
        itsLogger->Trace("TopRight does not match: " + boost::lexical_cast<std::string> (itsTopRight.X()) + " vs " + boost::lexical_cast<std::string> (other.TopRight().X()));
        itsLogger->Trace("TopRight does not match: " + boost::lexical_cast<std::string> (itsTopRight.Y()) + " vs " + boost::lexical_cast<std::string> (other.TopRight().Y()));
        return false;
    }

    if (itsProjection != other.Projection())
    {
        itsLogger->Trace("Projections don't match: " + boost::lexical_cast<std::string> (itsProjection) + " vs " + boost::lexical_cast<std::string> (other.Projection()));
        return false;
    }

    if (itsProjection == kRotatedLatLonProjection)
    {
		if (itsSouthPole != other.SouthPole())
    	{
        	itsLogger->Trace("SouthPole does not match: " + boost::lexical_cast<std::string> (itsSouthPole.X()) + " vs " + boost::lexical_cast<std::string> (other.SouthPole().X()));
        	itsLogger->Trace("SouthPole does not match: " + boost::lexical_cast<std::string> (itsSouthPole.Y()) + " vs " + boost::lexical_cast<std::string> (other.SouthPole().Y()));
        	return false;
    	}
    }

    if (itsProjection == kStereographicProjection)
    {
    	if (itsOrientation != other.Orientation())
    	{
    		itsLogger->Trace("Orientations don't match: " + boost::lexical_cast<std::string> (itsOrientation) + " vs " + boost::lexical_cast<std::string> (other.Orientation()));
    		return false;
    	}
    }

    if (Ni() != other.Ni())
    {
    	itsLogger->Trace("Grid X-counts don't match: " + boost::lexical_cast<std::string> (Ni()) + " vs " + boost::lexical_cast<std::string> (other.Ni()));
        return false;
    }

    if (Nj() != other.Nj())
    {
    	itsLogger->Trace("Grid Y-counts don't match: " + boost::lexical_cast<std::string> (Nj()) + " vs " + boost::lexical_cast<std::string> (other.Nj()));
        return false;
    }

    return true;

}

bool grid::operator!=(const grid& other) const
{
	return !(*this == other);
}

void grid::Data(std::shared_ptr<d_matrix_t> d)
{
	itsData = d;
}
