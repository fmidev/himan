#include "lambert_conformal_grid.h"
#include "info.h"
#include <ogr_spatialref.h>

using namespace himan;
using namespace std;

#ifdef HAVE_CUDA
#include "simple_packed.h"

// The following two functions are used for convenience in GPU specific code:
// in grid rotation we need the standard parallels and orientation, but in order
// to get those we need to include this file and ogr_spatialref.h, which is a lot
// of unnecessary code to be added to already slow compilation.

double GetStandardParallel(himan::grid* g, int parallelno)
{
	lambert_conformal_grid* lg = dynamic_cast<lambert_conformal_grid*>(g);

	if (lg)
	{
		if (parallelno == 1)
		{
			return lg->StandardParallel1();
		}
		else if (parallelno == 2)
		{
			return lg->StandardParallel2();
		}
	}

	return himan::kHPMissingValue;
}

double GetOrientation(himan::grid* g)
{
	lambert_conformal_grid* lg = dynamic_cast<lambert_conformal_grid*>(g);

	if (lg)
	{
		return lg->Orientation();
	}

	return kHPMissingValue;
}

#endif

lambert_conformal_grid::lambert_conformal_grid()
    : grid(kRegularGrid, kLambertConformalConic),
      itsBottomLeft(),
      itsTopLeft(),
      itsDi(kHPMissingValue),
      itsDj(kHPMissingValue),
      itsNi(kHPMissingInt),
      itsNj(kHPMissingInt),
      itsOrientation(kHPMissingValue),
      itsStandardParallel1(kHPMissingValue),
      itsStandardParallel2(kHPMissingValue),
      itsSouthPole(0, -90)
{
	itsLogger = logger("lambert_conformal_grid");
}

lambert_conformal_grid::lambert_conformal_grid(HPScanningMode theScanningMode, point theFirstPoint)
    : grid(kRegularGrid, kLambertConformalConic, theScanningMode),
      itsBottomLeft(),
      itsTopLeft(),
      itsDi(kHPMissingValue),
      itsDj(kHPMissingValue),
      itsNi(kHPMissingInt),
      itsNj(kHPMissingInt),
      itsOrientation(kHPMissingValue),
      itsStandardParallel1(kHPMissingValue),
      itsStandardParallel2(kHPMissingValue),
      itsSouthPole(0, -90)
{
	itsLogger = logger("lambert_conformal_grid");

	switch (theScanningMode)
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

lambert_conformal_grid::lambert_conformal_grid(const lambert_conformal_grid& other)
    : grid(other),
      itsBottomLeft(other.itsBottomLeft),
      itsTopLeft(other.itsTopLeft),
      itsDi(other.itsDi),
      itsDj(other.itsDj),
      itsNi(other.itsNi),
      itsNj(other.itsNj),
      itsOrientation(other.itsOrientation),
      itsStandardParallel1(other.itsStandardParallel1),
      itsStandardParallel2(other.itsStandardParallel2),
      itsSouthPole(other.itsSouthPole)
{
	itsLogger = logger("lambert_conformal_grid");
	SetCoordinates();  // Create transformer
}

lambert_conformal_grid::~lambert_conformal_grid() = default;

size_t lambert_conformal_grid::Size() const
{
	if (itsNi == static_cast<size_t>(kHPMissingInt) || itsNj == static_cast<size_t>(kHPMissingInt))
	{
		return kHPMissingInt;
	}

	return itsNi * itsNj;
}

point lambert_conformal_grid::TopRight() const
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

point lambert_conformal_grid::BottomLeft() const
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

point lambert_conformal_grid::TopLeft() const
{
	if (itsNi == static_cast<size_t>(kHPMissingInt) || itsNj == static_cast<size_t>(kHPMissingInt))
	{
		return point();
	}

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

point lambert_conformal_grid::BottomRight() const
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
	return itsTopLeft;
}

void lambert_conformal_grid::BottomLeft(const point& theBottomLeft)
{
	itsBottomLeft = theBottomLeft;
	itsTopLeft = point();  // "reset" other possible starting corner to
	                       // avoid situations where starting corner is changed
	                       // and old value persists

	SetCoordinates();
}

void lambert_conformal_grid::TopLeft(const point& theTopLeft)
{
	itsTopLeft = theTopLeft;
	itsBottomLeft = point();

	SetCoordinates();
}

point lambert_conformal_grid::FirstPoint() const
{
	switch (itsScanningMode)
	{
		case kTopLeft:
			return itsTopLeft;
		case kBottomLeft:
			return itsBottomLeft;
		default:
			throw runtime_error("Scanning mode not supported: " + HPScanningModeToString.at(itsScanningMode));
	}
}

point lambert_conformal_grid::LastPoint() const
{
	switch (itsScanningMode)
	{
		case kTopLeft:
			return BottomRight();
		case kBottomLeft:
			return TopRight();
		default:
			throw runtime_error("Scanning mode not supported: " + HPScanningModeToString.at(itsScanningMode));
	}
}

point lambert_conformal_grid::XY(const point& latlon) const
{
	if (!itsLatLonToXYTransformer)
	{
		if (!SetCoordinates())
		{
			return point();
		}
	}

	double projX = latlon.X(), projY = latlon.Y();
	assert(itsLatLonToXYTransformer);

	// 1. Transform latlon to projected coordinates.
	// Projected coordinates are in meters, with false easting and
	// false northing applied so that point 0,0 is top left or bottom left,
	// depending on the scanning mode.

	if (!itsLatLonToXYTransformer->Transform(1, &projX, &projY))
	{
		itsLogger.Error("Error determining xy value for latlon point " + boost::lexical_cast<std::string>(latlon.X()) +
		                "," + boost::lexical_cast<std::string>(latlon.Y()));
		return point();
	}

	// 2. Transform projected coordinates (meters) to grid xy (no unit).
	// Projected coordinates run from 0 ... area width and 0 ... area height.
	// Grid point coordinates run from 0 ... ni and 0 ... nj.

	const double x = (projX / itsDi);
	const double y = (projY / itsDj);

	return point(x, y);
}

point lambert_conformal_grid::LatLon(size_t locationIndex) const
{
	assert(itsNi != static_cast<size_t>(kHPMissingInt));
	assert(itsNj != static_cast<size_t>(kHPMissingInt));
	assert(!IsKHPMissingValue(Di()));
	assert(!IsKHPMissingValue(Dj()));
	assert(locationIndex < itsNi * itsNj);

	if (!itsXYToLatLonTransformer)
	{
		if (!SetCoordinates())
		{
			return point();
		}
	}

	const size_t jIndex = static_cast<size_t>(locationIndex / itsNi);
	const size_t iIndex = static_cast<size_t>(locationIndex % itsNi);

	double x = static_cast<double>(iIndex) * Di();
	double y = static_cast<double>(jIndex) * Dj();

	if (itsScanningMode == kTopLeft)
	{
		y *= -1;
	}

	assert(itsXYToLatLonTransformer);
	if (!itsXYToLatLonTransformer->Transform(1, &x, &y))
	{
		itsLogger.Error("Error determining latitude longitude value for xy point " +
		                boost::lexical_cast<std::string>(x) + "," + boost::lexical_cast<std::string>(y));
		return point();
	}

	return point(x, y);
}

bool lambert_conformal_grid::Swap(HPScanningMode newScanningMode)
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

void lambert_conformal_grid::SouthPole(const point& theSouthPole) { itsSouthPole = theSouthPole; }
point lambert_conformal_grid::SouthPole() const { return itsSouthPole; }
void lambert_conformal_grid::Ni(size_t theNi) { itsNi = theNi; }
void lambert_conformal_grid::Nj(size_t theNj) { itsNj = theNj; }
size_t lambert_conformal_grid::Ni() const { return itsNi; }
size_t lambert_conformal_grid::Nj() const { return itsNj; }
void lambert_conformal_grid::Di(double theDi) { itsDi = theDi; }
void lambert_conformal_grid::Dj(double theDj) { itsDj = theDj; }
void lambert_conformal_grid::Orientation(double theOrientation) { itsOrientation = theOrientation; }
double lambert_conformal_grid::Orientation() const { return itsOrientation; }
double lambert_conformal_grid::Di() const { return itsDi; }
double lambert_conformal_grid::Dj() const { return itsDj; }
bool lambert_conformal_grid::operator!=(const grid& other) const { return !(other == *this); }
bool lambert_conformal_grid::operator==(const grid& other) const
{
	const lambert_conformal_grid* g = dynamic_cast<const lambert_conformal_grid*>(&other);

	if (g)
	{
		return EqualsTo(*g);
	}

	return false;
}

bool lambert_conformal_grid::EqualsTo(const lambert_conformal_grid& other) const
{
	if (!grid::EqualsTo(other))
	{
		return false;
	}

	if (BottomLeft() != other.BottomLeft())
	{
		itsLogger.Trace("BottomLeft does not match: X " + boost::lexical_cast<std::string>(itsBottomLeft.X()) + " vs " +
		                boost::lexical_cast<std::string>(other.BottomLeft().X()));
		itsLogger.Trace("BottomLeft does not match: Y " + boost::lexical_cast<std::string>(itsBottomLeft.Y()) + " vs " +
		                boost::lexical_cast<std::string>(other.BottomLeft().Y()));
		return false;
	}

	if (TopLeft() != other.TopLeft())
	{
		itsLogger.Trace("BottomLeft does not match: X " + boost::lexical_cast<std::string>(itsBottomLeft.X()) + " vs " +
		                boost::lexical_cast<std::string>(other.BottomLeft().X()));
		itsLogger.Trace("BottomLeft does not match: Y " + boost::lexical_cast<std::string>(itsBottomLeft.Y()) + " vs " +
		                boost::lexical_cast<std::string>(other.BottomLeft().Y()));
		return false;
	}

	const double kEpsilon = 0.0001;

	if (fabs(itsDi - other.itsDi) > kEpsilon)
	{
		itsLogger.Trace("Di does not match: " + boost::lexical_cast<std::string>(Di()) + " vs " +
		                boost::lexical_cast<std::string>(other.Di()));
		return false;
	}

	if (fabs(itsDj - other.itsDj) > kEpsilon)
	{
		itsLogger.Trace("Dj does not match: " + boost::lexical_cast<std::string>(Dj()) + " vs " +
		                boost::lexical_cast<std::string>(other.Dj()));
		return false;
	}

	if (itsNi != other.Ni())
	{
		itsLogger.Trace("Ni does not match: " + boost::lexical_cast<std::string>(itsNi) + " vs " +
		                boost::lexical_cast<std::string>(other.Ni()));
		return false;
	}

	if (itsNj != other.Nj())
	{
		itsLogger.Trace("Nj does not match: " + boost::lexical_cast<std::string>(itsNj) + " vs " +
		                boost::lexical_cast<std::string>(other.Nj()));
		return false;
	}

	if (itsOrientation != other.itsOrientation)
	{
		itsLogger.Trace("Orientation does not match: " + boost::lexical_cast<std::string>(itsOrientation) + " vs " +
		                boost::lexical_cast<std::string>(other.itsOrientation));
		return false;
	}

	if (itsStandardParallel1 != other.itsStandardParallel1)
	{
		itsLogger.Trace(
		    "Standard latitude 1 does not match: " + boost::lexical_cast<std::string>(itsStandardParallel1) + " vs " +
		    boost::lexical_cast<std::string>(other.itsStandardParallel1));
		return false;
	}

	if (itsStandardParallel2 != other.itsStandardParallel2)
	{
		itsLogger.Trace(
		    "Standard latitude 2 does not match: " + boost::lexical_cast<std::string>(itsStandardParallel2) + " vs " +
		    boost::lexical_cast<std::string>(other.itsStandardParallel2));
		return false;
	}

	return true;
}

lambert_conformal_grid* lambert_conformal_grid::Clone() const { return new lambert_conformal_grid(*this); }
ostream& lambert_conformal_grid::Write(std::ostream& file) const
{
	grid::Write(file);

	file << itsBottomLeft << itsTopLeft << "__itsNi__ " << itsNi << endl
	     << "__itsNj__ " << itsNj << endl
	     << "__itsDi__ " << Di() << endl
	     << "__itsDj__ " << Dj() << endl
	     << "__itsOrientation__ " << itsOrientation << endl
	     << "__itsStandardParallel1__ " << itsStandardParallel1 << endl
	     << "__itsStandardParallel2__ " << itsStandardParallel2 << endl;

	return file;
}

void lambert_conformal_grid::StandardParallel1(double theStandardParallel1)
{
	itsStandardParallel1 = theStandardParallel1;
}

double lambert_conformal_grid::StandardParallel1() const { return itsStandardParallel1; }
void lambert_conformal_grid::StandardParallel2(double theStandardParallel2)
{
	itsStandardParallel2 = theStandardParallel2;
}

double lambert_conformal_grid::StandardParallel2() const { return itsStandardParallel2; }
bool lambert_conformal_grid::SetCoordinates() const
{
	itsSpatialReference = unique_ptr<OGRSpatialReference>(new OGRSpatialReference);

	// Build OGR presentation of LCC
	std::stringstream ss;

	if (IsKHPMissingValue(itsStandardParallel1) || IsKHPMissingValue(itsOrientation) || FirstPoint() == point())
	{
		// itsLogger.Error("First standard latitude or orientation missing");
		return false;
	}

	// If latin1==latin2, projection is effectively lccSP1

	if (IsKHPMissingValue(itsStandardParallel2))
	{
		itsStandardParallel2 = itsStandardParallel1;
	}

	ss << "+proj=lcc +lat_1=" << itsStandardParallel1 << " +lat_2=" << itsStandardParallel2
	   << " +lat_0=" << itsStandardParallel1 << " +lon_0=" << itsOrientation;

	// NOTE! Assuming earth is a sphere!
	ss << " +a=6367470 +b=6367470 +units=m +no_defs +wktext";

	auto err = itsSpatialReference->importFromProj4(ss.str().c_str());

	if (err != OGRERR_NONE)
	{
		itsLogger.Fatal("Error in area definition");
		abort();
	}

	// Area copy will be used for transform
	OGRSpatialReference* lccll = itsSpatialReference->CloneGeogCS();

	// Initialize transformer (latlon --> xy).
	// We need this to get east and north falsings because projection coordinates are
	// set for the area center and grid coordinates are in the area corner.

	itsLatLonToXYTransformer =
	    unique_ptr<OGRCoordinateTransformation>(OGRCreateCoordinateTransformation(lccll, itsSpatialReference.get()));

	assert(itsLatLonToXYTransformer);
	assert(itsScanningMode == kBottomLeft || itsScanningMode == kTopLeft);

	double falseEasting = FirstPoint().X();
	double falseNorthing = FirstPoint().Y();

	assert(!IsKHPMissingValue(falseEasting) && IsKHPMissingValue(falseNorthing));

	if (!itsLatLonToXYTransformer->Transform(1, &falseEasting, &falseNorthing))
	{
		itsLogger.Error("Error determining false easting and northing");
		return false;
	}

	// Setting falsings directly to translator will make handling them cleaner
	// later.

	ss << " +x_0=" << (-falseEasting) << " +y_0=" << (-falseNorthing);

	// itsLogger.Trace("PROJ4: " + ss.str());

	err = itsSpatialReference->importFromProj4(ss.str().c_str());

	if (err != OGRERR_NONE)
	{
		itsLogger.Error("Error in area definition");
		return false;
	}

	delete lccll;

	lccll = itsSpatialReference->CloneGeogCS();

	// Initialize transformer for later use (xy --> latlon))

	itsXYToLatLonTransformer = std::unique_ptr<OGRCoordinateTransformation>(
	    OGRCreateCoordinateTransformation(itsSpatialReference.get(), lccll));

	// ... and a transformer for reverse transformation

	itsLatLonToXYTransformer =
	    unique_ptr<OGRCoordinateTransformation>(OGRCreateCoordinateTransformation(lccll, itsSpatialReference.get()));

	delete lccll;

	return true;
}

OGRSpatialReference lambert_conformal_grid::SpatialReference() const
{
	return OGRSpatialReference(*itsSpatialReference);
}
