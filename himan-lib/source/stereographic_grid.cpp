#include "stereographic_grid.h"
#include <ogr_spatialref.h>

using namespace himan;
using namespace std;

stereographic_grid::stereographic_grid()
    : grid(kRegularGrid, kStereographic),
      itsBottomLeft(),
      itsTopLeft(),
      itsOrientation(kHPMissingInt),
      itsDi(kHPMissingValue),
      itsDj(kHPMissingValue),
      itsNi(kHPMissingInt),
      itsNj(kHPMissingInt)
{
	itsLogger = logger("stereographic_grid");
}

stereographic_grid::stereographic_grid(HPScanningMode theScanningMode, point theBottomLeft, point theTopLeft,
                                       double theOrientation)
    : grid(kRegularGrid, kStereographic, theScanningMode),
      itsBottomLeft(theBottomLeft),
      itsTopLeft(theTopLeft),
      itsOrientation(theOrientation)
{
	itsLogger = logger("stereographic_grid");
}

stereographic_grid::stereographic_grid(const stereographic_grid& other)
    : grid(other),
      itsBottomLeft(other.itsBottomLeft),
      itsTopLeft(other.itsTopLeft),
      itsOrientation(other.itsOrientation),
      itsDi(other.itsDi),
      itsDj(other.itsDj),
      itsNi(other.itsNi),
      itsNj(other.itsNj)
{
	itsLogger = logger("stereographic_grid");
}

stereographic_grid::~stereographic_grid() = default;

size_t stereographic_grid::Size() const
{
	if (itsNi == static_cast<size_t>(kHPMissingInt) || itsNj == static_cast<size_t>(kHPMissingInt))
	{
		return kHPMissingInt;
	}

	return itsNi * itsNj;
}

void stereographic_grid::Ni(size_t theNi)
{
	itsNi = theNi;
}
void stereographic_grid::Nj(size_t theNj)
{
	itsNj = theNj;
}
size_t stereographic_grid::Ni() const
{
	return itsNi;
}
size_t stereographic_grid::Nj() const
{
	return itsNj;
}
void stereographic_grid::Di(double theDi)
{
	itsDi = theDi;
}
void stereographic_grid::Dj(double theDj)
{
	itsDj = theDj;
}
double stereographic_grid::Di() const
{
	return itsDi;
}
double stereographic_grid::Dj() const
{
	return itsDj;
}
void stereographic_grid::Orientation(double theOrientation)
{
	itsOrientation = theOrientation;
}
double stereographic_grid::Orientation() const
{
	return itsOrientation;
}
HPScanningMode stereographic_grid::ScanningMode() const
{
	return itsScanningMode;
}
void stereographic_grid::ScanningMode(HPScanningMode theScanningMode)
{
	itsScanningMode = theScanningMode;
}

void stereographic_grid::CreateAreaAndGrid() const
{
	call_once(itsAreaFlag, [&]() {
		std::stringstream ss;

		if (itsOrientation == kHPMissingInt || itsEarthShape == earth_shape<double>() || FirstPoint() == point())
		{
			itsLogger.Fatal("Missing required area information");
			himan::Abort();
		}

		// see lambert_conformal.cpp for explanation for this function
		itsSpatialReference = unique_ptr<OGRSpatialReference>(new OGRSpatialReference);

		// clang-format off

		ss << "+proj=stere +lat_0=90 +lat_ts=60"
		   << " +lon_0=" << itsOrientation
		   << " +k=1 +units=m +no_defs"
		   << " +a=" << fixed << itsEarthShape.A()
		   << " +b=" << itsEarthShape.B()
		   << " +wktext";

		// clang-format on

		auto err = itsSpatialReference->importFromProj4(ss.str().c_str());

		if (err != OGRERR_NONE)
		{
			itsLogger.Fatal("Error in area definition");
			himan::Abort();
		}

		OGRSpatialReference* sterell = itsSpatialReference->CloneGeogCS();

		double falseEasting = FirstPoint().X();
		double falseNorthing = FirstPoint().Y();

		ASSERT(!IsKHPMissingValue(falseEasting) && !IsKHPMissingValue(falseNorthing));

		itsLatLonToXYTransformer = unique_ptr<OGRCoordinateTransformation>(
		    OGRCreateCoordinateTransformation(sterell, itsSpatialReference.get()));

		if (!itsLatLonToXYTransformer->Transform(1, &falseEasting, &falseNorthing))
		{
			itsLogger.Error("Error determining false easting and northing");
			himan::Abort();
		}

		// Setting falsings directly to translator will make handling them cleaner
		// later.

		ss << " +x_0=" << fixed << (-falseEasting) << " +y_0=" << (-falseNorthing);

		itsLogger.Trace(ss.str());

		err = itsSpatialReference->importFromProj4(ss.str().c_str());

		if (err != OGRERR_NONE)
		{
			itsLogger.Error("Error in area definition");
			himan::Abort();
		}

		delete sterell;

		sterell = itsSpatialReference->CloneGeogCS();

		// Initialize transformer for later use (xy --> latlon))

		itsXYToLatLonTransformer = std::unique_ptr<OGRCoordinateTransformation>(
		    OGRCreateCoordinateTransformation(itsSpatialReference.get(), sterell));

		// ... and a transformer for reverse transformation

		itsLatLonToXYTransformer = unique_ptr<OGRCoordinateTransformation>(
		    OGRCreateCoordinateTransformation(sterell, itsSpatialReference.get()));

		delete sterell;
	});
}

point stereographic_grid::XY(const point& latlon) const
{
	CreateAreaAndGrid();

	double projX = latlon.X(), projY = latlon.Y();

	if (!itsLatLonToXYTransformer->Transform(1, &projX, &projY))
	{
		itsLogger.Error("Error determining xy value for latlon point " + to_string(latlon.X()) + "," +
		                to_string(latlon.Y()));
		return point();
	}

	const double x = (projX / itsDi);
	const double y = (projY / itsDj);

	return point(x, y);
}

point stereographic_grid::LatLon(size_t locationIndex) const
{
	CreateAreaAndGrid();

	const size_t jIndex = static_cast<size_t>(locationIndex / itsNi);
	const size_t iIndex = static_cast<size_t>(locationIndex % itsNi);

	double x = static_cast<double>(iIndex) * Di();
	double y = static_cast<double>(jIndex) * Dj();

	if (itsScanningMode == kTopLeft)
	{
		y *= -1;
	}

	ASSERT(itsXYToLatLonTransformer);
	if (!itsXYToLatLonTransformer->Transform(1, &x, &y))
	{
		itsLogger.Error("Error determining latitude longitude value for xy point " + to_string(x) + "," + to_string(y));
		return point();
	}

	return point(x, y);
}

void stereographic_grid::BottomLeft(const point& theBottomLeft)
{
	itsBottomLeft = theBottomLeft;
}
void stereographic_grid::TopLeft(const point& theTopLeft)
{
	itsTopLeft = theTopLeft;
}
point stereographic_grid::BottomLeft() const
{
	switch (itsScanningMode)
	{
		case kBottomLeft:
			return itsBottomLeft;
		case kTopLeft:
			if (itsNi == static_cast<size_t>(kHPMissingInt) || itsNj == static_cast<size_t>(kHPMissingInt))
			{
				return point();
			}

			return LatLon(itsNj * itsNi - itsNi);
		case kUnknownScanningMode:
			return point();
		default:
			throw runtime_error("Unhandled scanning mode: " + HPScanningModeToString.at(itsScanningMode));
	}
}
point stereographic_grid::TopRight() const
{
	if (itsNi == static_cast<size_t>(kHPMissingInt) || itsNj == static_cast<size_t>(kHPMissingInt))
	{
		return point();
	}

	switch (itsScanningMode)
	{
		case kBottomLeft:
			return LatLon(itsNj * itsNi - 1);
		case kTopLeft:
			return LatLon(itsNi - 1);
		case kUnknownScanningMode:
			return point();
		default:
			throw runtime_error("Unhandled scanning mode: " + HPScanningModeToString.at(itsScanningMode));
	}
}
point stereographic_grid::TopLeft() const
{
	switch (itsScanningMode)
	{
		case kBottomLeft:
			if (itsNi == static_cast<size_t>(kHPMissingInt) || itsNj == static_cast<size_t>(kHPMissingInt))
			{
				return point();
			}

			return LatLon(itsNj * itsNi - itsNi);
		case kTopLeft:
			return itsTopLeft;
		case kUnknownScanningMode:
			return point();
		default:
			throw runtime_error("Unhandled scanning mode: " + HPScanningModeToString.at(itsScanningMode));
	}
}
point stereographic_grid::BottomRight() const
{
	if (itsNi == static_cast<size_t>(kHPMissingInt) || itsNj == static_cast<size_t>(kHPMissingInt))
	{
		return point();
	}

	switch (itsScanningMode)
	{
		case kBottomLeft:
			return LatLon(itsNi - 1);
		case kTopLeft:
			return LatLon(itsNi * itsNj - 1);
		case kUnknownScanningMode:
			return point();
		default:
			throw runtime_error("Unhandled scanning mode: " + HPScanningModeToString.at(itsScanningMode));
	}
}

point stereographic_grid::FirstPoint() const
{
	switch (itsScanningMode)
	{
		case kBottomLeft:
			return itsBottomLeft;
		case kTopLeft:
			return TopLeft();
		case kUnknownScanningMode:
			throw runtime_error("Scanning mode not set");
		default:
			throw runtime_error("Scanning mode not supported: " + HPScanningModeToString.at(itsScanningMode));
	}
}

void stereographic_grid::FirstPoint(const point& theFirstPoint)
{
	switch (itsScanningMode)
	{
		case kBottomLeft:
			itsBottomLeft = theFirstPoint;
			break;
		case kTopLeft:
			itsTopLeft = theFirstPoint;
			break;
		default:
			break;
	}
}

point stereographic_grid::LastPoint() const
{
	switch (itsScanningMode)
	{
		case kBottomLeft:
			return TopRight();
		case kTopLeft:
			return BottomRight();
		case kUnknownScanningMode:
			throw runtime_error("Scanning mode not set");
		default:
			throw runtime_error("Scanning mode not supported: " + HPScanningModeToString.at(itsScanningMode));
	}
}

bool stereographic_grid::Swap(HPScanningMode newScanningMode)
{
	if (itsScanningMode == newScanningMode)
	{
		return true;
	}

	// Flip with regards to x axis

	if ((itsScanningMode == kTopLeft && newScanningMode == kBottomLeft) ||
	    (itsScanningMode == kBottomLeft && newScanningMode == kTopLeft))
	{
		size_t halfSize = static_cast<size_t>(floor(Nj() / 2));

		for (size_t y = 0; y < halfSize; y++)
		{
			for (size_t x = 0; x < Ni(); x++)
			{
				double upper = itsData.At(x, y);
				double lower = itsData.At(x, Nj() - 1 - y);

				itsData.Set(x, y, 0, lower);
				itsData.Set(x, Nj() - 1 - y, 0, upper);
			}
		}
	}
	else
	{
		itsLogger.Error("Swap from mode " + string(HPScanningModeToString.at(itsScanningMode)) + " to mode " +
		                string(HPScanningModeToString.at(newScanningMode)) + " not implemented yet");
		return false;
	}

	itsScanningMode = newScanningMode;

	return true;
}

stereographic_grid* stereographic_grid::Clone() const
{
	return new stereographic_grid(*this);
}
ostream& stereographic_grid::Write(std::ostream& file) const
{
	grid::Write(file);

	file << itsBottomLeft;
	file << itsTopLeft;
	file << "__itsNi__ " << itsNi << endl;
	file << "__itsNj__ " << itsNj << endl;
	file << "__itsDi__ " << itsDi << endl;
	file << "__itsDj__ " << itsDj << endl;

	file << "__itsOrientation__ " << itsOrientation << endl;

	return file;
}

bool stereographic_grid::operator!=(const grid& other) const
{
	return !(other == *this);
}
bool stereographic_grid::operator==(const grid& other) const
{
	const stereographic_grid* g = dynamic_cast<const stereographic_grid*>(&other);

	if (g)
	{
		return EqualsTo(*g);
	}

	return false;
}

bool stereographic_grid::EqualsTo(const stereographic_grid& other) const
{
	if (!grid::EqualsTo(other))
	{
		return false;
	}

	if (BottomLeft() != other.BottomLeft())
	{
		itsLogger.Trace("BottomLeft does not match: X " + to_string(BottomLeft().X()) + " vs " +
		                to_string(other.BottomLeft().X()));
		itsLogger.Trace("BottomLeft does not match: Y " + to_string(BottomLeft().Y()) + " vs " +
		                to_string(other.BottomLeft().Y()));
		return false;
	}

	if (TopLeft() != other.TopLeft())
	{
		itsLogger.Trace("TopLeft does not match: X " + to_string(TopLeft().X()) + " vs " +
		                to_string(other.TopLeft().X()));
		itsLogger.Trace("TopLeft does not match: Y " + to_string(TopLeft().Y()) + " vs " +
		                to_string(other.TopLeft().Y()));
		return false;
	}

	if (TopRight() != other.TopRight())
	{
		itsLogger.Trace("TopRight does not match: X " + to_string(TopRight().X()) + " vs " +
		                to_string(other.TopRight().X()));
		itsLogger.Trace("TopRight does not match: Y " + to_string(TopRight().Y()) + " vs " +
		                to_string(other.TopRight().Y()));
		return false;
	}

	if (itsDi != other.Di())
	{
		itsLogger.Trace("Di does not match: " + to_string(itsDi) + " vs " + to_string(other.Di()));
		return false;
	}

	if (itsDj != other.Dj())
	{
		itsLogger.Trace("Dj does not match: " + to_string(itsDj) + " vs " + to_string(other.Dj()));
		return false;
	}

	if (itsNi != other.Ni())
	{
		itsLogger.Trace("Ni does not match: " + to_string(itsNi) + " vs " + to_string(other.Ni()));
		return false;
	}

	if (itsNj != other.Nj())
	{
		itsLogger.Trace("Nj does not match: " + to_string(itsNj) + " vs " + to_string(other.Nj()));
		return false;
	}

	if (itsOrientation != other.Orientation())
	{
		itsLogger.Trace("Orientation does not match: " + to_string(itsOrientation) + " vs " +
		                to_string(other.Orientation()));
		return false;
	}

	return true;
}
