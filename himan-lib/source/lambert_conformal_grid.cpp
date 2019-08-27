#include "lambert_conformal_grid.h"
#include "info.h"
#include <functional>
#include <ogr_spatialref.h>

using namespace himan;
using namespace std;

#ifdef HAVE_CUDA
// The following two functions are used for convenience in GPU specific code:
// in grid rotation we need the standard parallels and orientation, but in order
// to get those we need to include this file and ogr_spatialref.h, which is a
// lot
// of unnecessary code to be added to already slow compilation.

double GetOrientation(const himan::grid* g)
{
	const auto lg = dynamic_cast<const lambert_conformal_grid*>(g);

	if (lg)
	{
		return lg->Orientation();
	}

	return MissingDouble();
}

double GetCone(const himan::grid* g)
{
	const auto lg = dynamic_cast<const lambert_conformal_grid*>(g);

	if (lg)
	{
		return lg->Cone();
	}

	return MissingDouble();
}

#endif

lambert_conformal_grid::lambert_conformal_grid()
    : regular_grid(),
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
	Type(kLambertConformalConic);
}

lambert_conformal_grid::lambert_conformal_grid(HPScanningMode theScanningMode, point theFirstPoint)
    : regular_grid(),
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
	Type(kLambertConformalConic);
	ScanningMode(theScanningMode);
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
    : regular_grid(other),
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
}

void lambert_conformal_grid::TopLeft(const point& theTopLeft)
{
	itsTopLeft = theTopLeft;
	itsBottomLeft = point();
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
	SetCoordinates();

	double projX = latlon.X(), projY = latlon.Y();
	ASSERT(itsLatLonToXYTransformer);

	// 1. Transform latlon to projected coordinates.
	// Projected coordinates are in meters, with false easting and
	// false northing applied so that point 0,0 is top left or bottom left,
	// depending on the scanning mode.

	if (!itsLatLonToXYTransformer->Transform(1, &projX, &projY))
	{
		itsLogger.Error("Error determining xy value for latlon point " + to_string(latlon.X()) + "," +
		                to_string(latlon.Y()));
		return point();
	}

	// 2. Transform projected coordinates (meters) to grid xy (no unit).
	// Projected coordinates run from 0 ... area width and 0 ... area height.
	// Grid point coordinates run from 0 ... ni and 0 ... nj.

	const double x = (projX / itsDi);
	const double y = (projY / itsDj);

	if (x < 0 || x > Ni() - 1 || y < 0 || y > Nj() - 1)
	{
		return point(MissingDouble(), MissingDouble());
	}

	return point(x, y);
}

point lambert_conformal_grid::LatLon(size_t locationIndex) const
{
	ASSERT(itsNi != static_cast<size_t>(kHPMissingInt));
	ASSERT(itsNj != static_cast<size_t>(kHPMissingInt));
	ASSERT(!IsKHPMissingValue(Di()));
	ASSERT(!IsKHPMissingValue(Dj()));
	ASSERT(locationIndex < itsNi * itsNj);

	SetCoordinates();

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

size_t lambert_conformal_grid::Hash() const
{
	vector<size_t> hashes;
	hashes.push_back(Type());
	hashes.push_back(FirstPoint().Hash());
	hashes.push_back(Ni());
	hashes.push_back(Nj());
	hashes.push_back(hash<double>{}(Di()));
	hashes.push_back(hash<double>{}(Dj()));
	hashes.push_back(ScanningMode());
	hashes.push_back(hash<double>{}(Orientation()));
	hashes.push_back(hash<double>{}(StandardParallel1()));
	hashes.push_back(hash<double>{}(StandardParallel2()));
	return boost::hash_range(hashes.begin(), hashes.end());
}

void lambert_conformal_grid::SouthPole(const point& theSouthPole)
{
	itsSouthPole = theSouthPole;
}
point lambert_conformal_grid::SouthPole() const
{
	return itsSouthPole;
}
void lambert_conformal_grid::Ni(size_t theNi)
{
	itsNi = theNi;
}
void lambert_conformal_grid::Nj(size_t theNj)
{
	itsNj = theNj;
}
size_t lambert_conformal_grid::Ni() const
{
	return itsNi;
}
size_t lambert_conformal_grid::Nj() const
{
	return itsNj;
}
void lambert_conformal_grid::Di(double theDi)
{
	itsDi = theDi;
}
void lambert_conformal_grid::Dj(double theDj)
{
	itsDj = theDj;
}
void lambert_conformal_grid::Orientation(double theOrientation)
{
	itsOrientation = theOrientation;
}
double lambert_conformal_grid::Orientation() const
{
	return itsOrientation;
}
double lambert_conformal_grid::Di() const
{
	return itsDi;
}
double lambert_conformal_grid::Dj() const
{
	return itsDj;
}
bool lambert_conformal_grid::operator!=(const grid& other) const
{
	return !(other == *this);
}
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
	if (!regular_grid::EqualsTo(other))
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

	const double kEpsilon = 0.0001;

	if (fabs(itsDi - other.itsDi) > kEpsilon)
	{
		itsLogger.Trace("Di does not match: " + to_string(Di()) + " vs " + to_string(other.Di()));
		return false;
	}

	if (fabs(itsDj - other.itsDj) > kEpsilon)
	{
		itsLogger.Trace("Dj does not match: " + to_string(Dj()) + " vs " + to_string(other.Dj()));
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

	if (itsOrientation != other.itsOrientation)
	{
		itsLogger.Trace("Orientation does not match: " + to_string(itsOrientation) + " vs " +
		                to_string(other.itsOrientation));
		return false;
	}

	if (itsStandardParallel1 != other.itsStandardParallel1)
	{
		itsLogger.Trace("Standard latitude 1 does not match: " + to_string(itsStandardParallel1) + " vs " +
		                to_string(other.itsStandardParallel1));
		return false;
	}

	if (itsStandardParallel2 != other.itsStandardParallel2)
	{
		itsLogger.Trace("Standard latitude 2 does not match: " + to_string(itsStandardParallel2) + " vs " +
		                to_string(other.itsStandardParallel2));
		return false;
	}

	return true;
}

unique_ptr<grid> lambert_conformal_grid::Clone() const
{
	return unique_ptr<grid>(new lambert_conformal_grid(*this));
}

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

double lambert_conformal_grid::StandardParallel1() const
{
	return itsStandardParallel1;
}
void lambert_conformal_grid::StandardParallel2(double theStandardParallel2)
{
	itsStandardParallel2 = theStandardParallel2;
}

double lambert_conformal_grid::StandardParallel2() const
{
	return itsStandardParallel2;
}
void lambert_conformal_grid::SetCoordinates() const
{
	call_once(itsAreaFlag, [&]() {
		itsSpatialReference = unique_ptr<OGRSpatialReference>(new OGRSpatialReference);

		// Build OGR presentation of LCC
		std::stringstream ss;

		if (IsKHPMissingValue(itsStandardParallel1) || IsKHPMissingValue(itsOrientation) || FirstPoint() == point())
		{
			// itsLogger.Error("First standard latitude or orientation missing");
			return;
		}

		// If itsStandardParallel1==itsStandardParallel2, projection is effectively lccSP1

		if (IsKHPMissingValue(itsStandardParallel2))
		{
			itsStandardParallel2 = itsStandardParallel1;
		}

		// clang-format off

		ss << "+proj=lcc +lat_1=" << itsStandardParallel1
		   << " +lat_2=" << itsStandardParallel2
		   << " +lat_0=" << itsStandardParallel1
		   << " +lon_0=" << itsOrientation
		   << " +a=" << fixed << itsEarthShape.A()
		   << " +b=" << itsEarthShape.B()
		   << " +units=m +no_defs +wktext";

		// clang-format on

		auto err = itsSpatialReference->importFromProj4(ss.str().c_str());

		if (err != OGRERR_NONE)
		{
			itsLogger.Fatal("Error in area definition");
			himan::Abort();
		}

		// Area copy will be used for transform
		OGRSpatialReference* lccll = itsSpatialReference->CloneGeogCS();

		// Initialize transformer (latlon --> xy).
		// We need this to get east and north falsings because projection
		// coordinates are
		// set for the area center and grid coordinates are in the area corner.

		itsLatLonToXYTransformer = unique_ptr<OGRCoordinateTransformation>(
		    OGRCreateCoordinateTransformation(lccll, itsSpatialReference.get()));

		ASSERT(itsLatLonToXYTransformer);
		ASSERT(itsScanningMode == kBottomLeft || itsScanningMode == kTopLeft);

		const point fp = FirstPoint();
		double falseEasting = fp.X();
		double falseNorthing = fp.Y();

		ASSERT(!IsKHPMissingValue(falseEasting) && !IsKHPMissingValue(falseNorthing));

		if (!itsLatLonToXYTransformer->Transform(1, &falseEasting, &falseNorthing))
		{
			itsLogger.Error("Error determining false easting and northing");
			return;
		}

		// Setting falsings directly to translator will make handling them cleaner
		// later.

		ss << " +x_0=" << fixed << (-falseEasting) << " +y_0=" << (-falseNorthing);

		itsLogger.Trace(ss.str());

		err = itsSpatialReference->importFromProj4(ss.str().c_str());

		if (err != OGRERR_NONE)
		{
			itsLogger.Error("Error in area definition");
			return;
		}

		delete lccll;

		lccll = itsSpatialReference->CloneGeogCS();

		// Initialize transformer for later use (xy --> latlon))

		itsXYToLatLonTransformer = std::unique_ptr<OGRCoordinateTransformation>(
		    OGRCreateCoordinateTransformation(itsSpatialReference.get(), lccll));

		// ... and a transformer for reverse transformation

		itsLatLonToXYTransformer = unique_ptr<OGRCoordinateTransformation>(
		    OGRCreateCoordinateTransformation(lccll, itsSpatialReference.get()));

		delete lccll;
	});
}

OGRSpatialReference lambert_conformal_grid::SpatialReference() const
{
	SetCoordinates();
	return OGRSpatialReference(*itsSpatialReference);
}

double lambert_conformal_grid::Cone() const
{
	if (fabs(itsStandardParallel1 - itsStandardParallel2) < 0.0001)
	{
		return sin(fabs(itsStandardParallel1) * constants::kDeg);
	}

	return (log(cos(itsStandardParallel1 * constants::kDeg)) - log(cos(itsStandardParallel2 * constants::kDeg))) /
	       (log(tan((90 - fabs(itsStandardParallel1)) * constants::kDeg * 0.5)) -
	        log(tan(90 - fabs(itsStandardParallel2)) * constants::kDeg * 0.5));
}
