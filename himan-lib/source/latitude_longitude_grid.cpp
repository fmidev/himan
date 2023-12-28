#include "latitude_longitude_grid.h"
#include "info.h"
#include <functional>
#include <ogr_spatialref.h>

using namespace himan;
using namespace std;

latitude_longitude_grid::latitude_longitude_grid(HPScanningMode theScanningMode, const point& theFirstPoint, size_t ni,
                                                 size_t nj, double di, double dj, const earth_shape<double>& earthShape,
                                                 const std::string& theName)
    : regular_grid(kLatitudeLongitude, theScanningMode, di, dj, ni, nj, false, theName), itsFirstPoint(theFirstPoint)
{
	itsLogger = logger("latitude_longitude_grid");
	itsEarthShape = earthShape;
	itsLogger.Trace(Proj4String());
}

latitude_longitude_grid::latitude_longitude_grid(HPScanningMode theScanningMode, const point& theFirstPoint,
                                                 const point& theLastPoint, size_t ni, size_t nj,
                                                 const earth_shape<double>& earthShape, const std::string& theName)
    : regular_grid(kLatitudeLongitude, theScanningMode, MissingDouble(), MissingDouble(), ni, nj, false, theName),
      itsFirstPoint(theFirstPoint)
{
	double fx = theFirstPoint.X();
	while (fx > theLastPoint.X())
		fx -= 360;

	itsDi = fabs(theLastPoint.X() - fx) / static_cast<double>(ni - 1);
	itsDj = fabs(theLastPoint.Y() - theFirstPoint.Y()) / static_cast<double>(nj - 1);
	itsEarthShape = earthShape;
	itsLogger = logger("latitude_longitude_grid");
	itsLogger.Trace(Proj4String());
}

latitude_longitude_grid::latitude_longitude_grid(const latitude_longitude_grid& other)
    : regular_grid(other), itsFirstPoint(other.itsFirstPoint), itsEarthShape(other.itsEarthShape)
{
	itsLogger = logger("latitude_longitude_grid");
}

point latitude_longitude_grid::FirstPoint() const
{
	return itsFirstPoint;
}

earth_shape<double> latitude_longitude_grid::EarthShape() const
{
	return itsEarthShape;
}

bool latitude_longitude_grid::IsGlobal() const
{
	if (static_cast<double>(Ni()) * Di() == 360.0)
		return true;
	return false;
}

std::vector<point> latitude_longitude_grid::GridPointsInProjectionSpace() const
{
	std::vector<point> ret;
	ret.reserve(Size());

	point first = Projected(FirstPoint());

	first.X(std::stod(fmt::format("{:.7f}", first.X())));
	first.Y(std::stod(fmt::format("{:.7f}", first.Y())));

	const double dj = itsDj * (itsScanningMode == kTopLeft ? -1 : 1);

	for (size_t y = 0; y < Nj(); y++)
	{
		for (size_t x = 0; x < Ni(); x++)
		{
			ret.emplace_back(fma(static_cast<double>(x), itsDi, first.X()), fma(static_cast<double>(y), dj, first.Y()));
		}
	}

	return ret;
}

std::vector<point> latitude_longitude_grid::XY(const regular_grid& target) const
{
	// Almost the same function as regular_grid::XY(), but we here we rely
	// on class function XY(point) to check the grid containment
	// and do not utilize OGRCoordinateTransform class

	// 1. Create list of points in the projection space of the
	// target grid.

	const auto targetProj = target.GridPointsInProjectionSpace();

	// 2. Transform the points to the projection space of the source
	// grid
	// 3. Transform projected coordinates to grid space

	auto targetsp = target.SpatialReference();
	auto sp = std::unique_ptr<OGRSpatialReference>(new OGRSpatialReference);
	sp->importFromProj4(Proj4String().c_str());

	vector<point> sourceXY;
	sourceXY.reserve(targetProj.size());

	if (sp->IsSame(targetsp.get()))
	{
		itsLogger.Trace("Spatial references are equal, no need to do transformation");

		for (const auto& p : targetProj)
		{
			sourceXY.push_back(XY(p));
		}
	}
	else
	{
		for (const auto& p : targetProj)
		{
			sourceXY.push_back(XY(target.LatLon(p)));
		}
	}

	return sourceXY;
}

point latitude_longitude_grid::XY(const himan::point& latlon) const
{
	ASSERT(itsScanningMode == kTopLeft || itsScanningMode == kBottomLeft);

	double x = (latlon.X() - itsFirstPoint.X()) / Di();

	if (x < 0 || x > static_cast<double>(Ni() - 1))
	{
		if (IsGlobal())
		{
			// wrap x if necessary
			// this might happen f.ex. with EC where grid start at 0 meridian and
			// we interpolate from say -10 to 40 longitude

			while (x < 0)
			{
				x += static_cast<double>(Ni());
			}
			while (x > static_cast<double>(Ni()))
			{
				x -= static_cast<double>(Ni());
			}
		}
		else
		{
			x = MissingDouble();
		}
	}

	double y;

	switch (itsScanningMode)
	{
		case kBottomLeft:
			y = (latlon.Y() - itsFirstPoint.Y()) / Dj();
			break;
		case kTopLeft:
			y = (itsFirstPoint.Y() - latlon.Y()) / Dj();
			break;
		default:
			itsLogger.Fatal("Scanning mode not supported: " + HPScanningModeToString.at(itsScanningMode));
			himan::Abort();
	}

	if (y < 0 || y > static_cast<double>(Nj()) - 1)
	{
		y = MissingDouble();
	}

	return point(x, y);
}

point latitude_longitude_grid::LatLon(const point& projected) const
{
	return projected;
}

point latitude_longitude_grid::LatLon(size_t locationIndex) const
{
	ASSERT(locationIndex < Size());

	const double j = floor(static_cast<double>(locationIndex / Ni()));
	const double i = fmod(static_cast<double>(locationIndex), static_cast<double>(Ni()));

	point ret(itsFirstPoint.X() + i * Di(), kHPMissingInt);

	switch (itsScanningMode)
	{
		case kBottomLeft:
			ret.Y(itsFirstPoint.Y() + j * Dj());
			break;

		case kTopLeft:
			ret.Y(itsFirstPoint.Y() - j * Dj());
			break;
		default:
			throw runtime_error("Scanning mode not supported: " + HPScanningModeToString.at(itsScanningMode));
	}

	while (ret.X() >= 360.)
	{
		ret.X(ret.X() - 360.);
	}
	if (Nj() == 1)
	{
		ret.Y(itsFirstPoint.Y());
	}
	return ret;
}

std::string latitude_longitude_grid::Proj4String() const
{
	return fmt::format("+proj=longlat {} +no_defs", itsEarthShape.Proj4String());
}

std::unique_ptr<OGRSpatialReference> latitude_longitude_grid::SpatialReference() const
{
	auto sp = std::unique_ptr<OGRSpatialReference>(new OGRSpatialReference());
	sp->importFromProj4(Proj4String().c_str());
	return std::move(sp);
}

point latitude_longitude_grid::Projected(const point& latlon) const
{
	return latlon;
}
size_t latitude_longitude_grid::Hash() const
{
	vector<size_t> hashes;
	hashes.push_back(Type());
	hashes.push_back(FirstPoint().Hash());
	hashes.push_back(Ni());
	hashes.push_back(Nj());
	hashes.push_back(hash<double>{}(Di()));
	hashes.push_back(hash<double>{}(Dj()));
	hashes.push_back(ScanningMode());
	return boost::hash_range(hashes.begin(), hashes.end());
}

bool latitude_longitude_grid::operator!=(const grid& other) const
{
	return !(other == *this);
}
bool latitude_longitude_grid::operator==(const grid& other) const
{
	const latitude_longitude_grid* g = dynamic_cast<const latitude_longitude_grid*>(&other);

	if (g)
	{
		return EqualsTo(*g);
	}

	return false;
}

bool latitude_longitude_grid::EqualsTo(const latitude_longitude_grid& other) const
{
	if (!regular_grid::EqualsTo(other))
	{
		return false;
	}

	return true;
}

unique_ptr<grid> latitude_longitude_grid::Clone() const
{
	return unique_ptr<grid>(new latitude_longitude_grid(*this));
}

ostream& latitude_longitude_grid::Write(std::ostream& file) const
{
	regular_grid::Write(file);

	file << "__itsIsGlobal__" << IsGlobal() << endl;

	return file;
}

rotated_latitude_longitude_grid::rotated_latitude_longitude_grid(
    HPScanningMode theScanningMode, const point& theFirstPoint, size_t ni, size_t nj, double di, double dj,
    const earth_shape<double>& earthShape, const point& theSouthPole, bool initiallyRotated, const std::string& theName)
    : latitude_longitude_grid(theScanningMode, theFirstPoint, ni, nj, di, dj, earthShape, theName),
      itsSouthPole(theSouthPole)
{
	if (!initiallyRotated)
	{
		throw std::runtime_error("Unable to create rotated_latitude_longitude_grid with unrotated coordinates, yet");
	}

	itsFromRotLatLon = himan::geoutil::rotation<double>().FromRotLatLon(theSouthPole.Y() * constants::kDeg,
	                                                                    theSouthPole.X() * constants::kDeg, 0);
	itsToRotLatLon = himan::geoutil::rotation<double>().ToRotLatLon(theSouthPole.Y() * constants::kDeg,
	                                                                theSouthPole.X() * constants::kDeg, 0);

	itsGridType = kRotatedLatitudeLongitude;
	itsLogger = logger("rotated_latitude_longitude_grid");
	itsLogger.Trace(Proj4String());
}

rotated_latitude_longitude_grid::rotated_latitude_longitude_grid(
    HPScanningMode theScanningMode, const point& theFirstPoint, const point& theLastPoint, size_t ni, size_t nj,
    const earth_shape<double>& earthShape, const point& theSouthPole, bool initiallyRotated, const std::string& theName)
    : latitude_longitude_grid(theScanningMode, theFirstPoint, theLastPoint, ni, nj, earthShape, theName),
      itsSouthPole(theSouthPole)
{
	if (!initiallyRotated)
	{
		throw std::runtime_error("Unable to create rotated_latitude_longitude_grid with unrotated coordinates, yet");
	}

	itsFromRotLatLon = himan::geoutil::rotation<double>().FromRotLatLon(theSouthPole.Y() * constants::kDeg,
	                                                                    theSouthPole.X() * constants::kDeg, 0);
	itsToRotLatLon = himan::geoutil::rotation<double>().ToRotLatLon(theSouthPole.Y() * constants::kDeg,
	                                                                theSouthPole.X() * constants::kDeg, 0);

	itsGridType = kRotatedLatitudeLongitude;
	itsLogger = logger("rotated_latitude_longitude_grid");
	itsLogger.Trace(Proj4String());
}

rotated_latitude_longitude_grid::rotated_latitude_longitude_grid(const rotated_latitude_longitude_grid& other)
    : latitude_longitude_grid(other),
      itsSouthPole(other.itsSouthPole),
      itsFromRotLatLon(other.itsFromRotLatLon),
      itsToRotLatLon(other.itsToRotLatLon)
{
	itsLogger = logger("rotated_latitude_longitude_grid");
}

bool rotated_latitude_longitude_grid::operator!=(const grid& other) const
{
	return !(other == *this);
}
bool rotated_latitude_longitude_grid::operator==(const grid& other) const
{
	const rotated_latitude_longitude_grid* g = dynamic_cast<const rotated_latitude_longitude_grid*>(&other);

	if (g)
	{
		return EqualsTo(*g);
	}

	return false;
}

bool rotated_latitude_longitude_grid::EqualsTo(const rotated_latitude_longitude_grid& other) const
{
	if (!latitude_longitude_grid::EqualsTo(other))
	{
		return false;
	}

	if (!point::LatLonCompare(itsSouthPole, other.SouthPole()))
	{
		itsLogger.Trace(fmt::format("SouthPole does not match: X {} vs {}", itsSouthPole.X(), other.SouthPole().X()));
		itsLogger.Trace(fmt::format("SouthPole does not match: Y {} vs {}", itsSouthPole.Y(), other.SouthPole().Y()));
		return false;
	}

	return true;
}

unique_ptr<grid> rotated_latitude_longitude_grid::Clone() const
{
	return unique_ptr<grid>(new rotated_latitude_longitude_grid(*this));
}

point rotated_latitude_longitude_grid::SouthPole() const
{
	return itsSouthPole;
}

point rotated_latitude_longitude_grid::FirstPoint() const
{
	return LatLon(0);
}

point rotated_latitude_longitude_grid::Rotate(const point& latlon) const
{
	himan::geoutil::position<double> p(latlon.Y() * constants::kDeg, latlon.X() * constants::kDeg, 0.0,
	                                   earth_shape<double>(1.0));
	himan::geoutil::rotate(p, itsToRotLatLon);

	return point(p.Lon(earth_shape<double>(1.0)) * constants::kRad, p.Lat(earth_shape<double>(1.0)) * constants::kRad);
}
point rotated_latitude_longitude_grid::XY(const point& latlon) const
{
	return latitude_longitude_grid::XY(Rotate(latlon));
}

point rotated_latitude_longitude_grid::LatLon(const point& rotated) const
{
	himan::geoutil::position<double> p(rotated.Y() * constants::kDeg, rotated.X() * constants::kDeg, 0.0,
	                                   earth_shape<double>(1.0));
	himan::geoutil::rotate(p, itsFromRotLatLon);

	return point(p.Lon(earth_shape<double>(1.0)) * constants::kRad, p.Lat(earth_shape<double>(1.0)) * constants::kRad);
}

point rotated_latitude_longitude_grid::LatLon(size_t locationIndex) const
{
	point rll = latitude_longitude_grid::LatLon(locationIndex);  // rotated coordinates

	himan::geoutil::position<double> p(rll.Y() * constants::kDeg, rll.X() * constants::kDeg, 0.0,
	                                   earth_shape<double>(1.0));
	himan::geoutil::rotate(p, itsFromRotLatLon);
	return point(p.Lon(earth_shape<double>(1.0)) * constants::kRad, p.Lat(earth_shape<double>(1.0)) * constants::kRad);
}

point rotated_latitude_longitude_grid::RotatedLatLon(size_t locationIndex) const
{
	return latitude_longitude_grid::LatLon(locationIndex);
}

ostream& rotated_latitude_longitude_grid::Write(std::ostream& file) const
{
	latitude_longitude_grid::Write(file);
	file << itsSouthPole;

	return file;
}

size_t rotated_latitude_longitude_grid::Hash() const
{
	vector<size_t> hashes;
	hashes.push_back(Type());
	hashes.push_back(FirstPoint().Hash());
	hashes.push_back(Ni());
	hashes.push_back(Nj());
	hashes.push_back(hash<double>{}(Di()));
	hashes.push_back(hash<double>{}(Dj()));
	hashes.push_back(ScanningMode());
	hashes.push_back(SouthPole().Hash());
	return boost::hash_range(hashes.begin(), hashes.end());
}

std::string rotated_latitude_longitude_grid::Proj4String() const
{
	// navan longitudi menee +lon_0 asetukseen ja +o_lon_p on aina nolla
	// longitudi asetetaan nollaksi, koska uuden koordinaadiston nollameridiaani
	// asetetaan kierrettyyn kohtaan
	// o_lon_p on longitudissa nolla uudessa kierretyssä koordinaatistossa, jonka lon_0 asettaa
	// Hirlamilla kiertokulma on 0
	// Tuurilla ne laivatkin seilaa

	return fmt::format(
	    "+proj=ob_tran +o_proj=longlat +lon_0={} +o_lon_p=0 +o_lat_p={} {} +to_meter=0.0174532925199 "
	    "+no_defs",
	    itsSouthPole.X(), (itsSouthPole.Y() * -1), itsEarthShape.Proj4String());
}

std::unique_ptr<OGRSpatialReference> rotated_latitude_longitude_grid::SpatialReference() const
{
	auto sp = std::unique_ptr<OGRSpatialReference>(new OGRSpatialReference());
	sp->importFromProj4(Proj4String().c_str());
	return std::move(sp);
}

std::vector<point> rotated_latitude_longitude_grid::XY(const regular_grid& target) const
{
	const auto targetProj = target.GridPointsInProjectionSpace();

	// 2. Transform the points to the projection space of the source
	// grid
	// 3. Transform projected coordinates to grid space

	auto targetsp = target.SpatialReference();
	auto sp = std::unique_ptr<OGRSpatialReference>(new OGRSpatialReference);
	sp->importFromProj4(Proj4String().c_str());

	vector<point> sourceXY;
	sourceXY.reserve(targetProj.size());

	if (sp->IsSame(targetsp.get()))
	{
		itsLogger.Trace("Spatial references are equal, no need to do transformation");

		for (const auto& p : targetProj)
		{
			sourceXY.push_back(latitude_longitude_grid::XY(p));
		}
	}
	else
	{
		for (const auto& p : targetProj)
		{
			sourceXY.push_back(XY(target.LatLon(p)));
		}
	}

	return sourceXY;
}

point rotated_latitude_longitude_grid::Projected(const point& latlon) const
{
	return Rotate(latlon);
}
