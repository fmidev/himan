#include "geostationary_grid.h"
#include <boost/functional/hash.hpp>
#include <cmath>
#include <ogr_spatialref.h>

using namespace himan;

namespace
{
double NormalizeLongitude(double lon)
{
	while (lon > 180.)
	{
		lon -= 360.;
	}
	while (lon < -180.)
	{
		lon += 360.;
	}
	return lon;
}
}  // namespace

geostationary_grid::geostationary_grid(HPScanningMode theScanningMode, size_t ni, size_t nj, long nr, long dx, long dy,
                                       long xp, long yp, long subSatelliteLongitude, long subSatelliteLatitude,
                                       const earth_shape<double>& earthShape, char sweepAxis,
                                       const std::string& theName)
    : regular_grid(kGeostationary, theScanningMode, MissingDouble(), MissingDouble(), ni, nj, false, theName),
      itsNr(nr),
      itsDx(dx),
      itsDy(dy),
      itsXp(xp),
      itsYp(yp),
      itsSubSatelliteLongitude(subSatelliteLongitude),
      itsSubSatelliteLatitude(subSatelliteLatitude),
      itsSweepAxis(sweepAxis)
{
	itsLogger = logger("geostationary_grid");
	Init(earthShape);
}

geostationary_grid::geostationary_grid(const geostationary_grid& other)
    : regular_grid(other),
      itsNr(other.itsNr),
      itsDx(other.itsDx),
      itsDy(other.itsDy),
      itsXp(other.itsXp),
      itsYp(other.itsYp),
      itsSubSatelliteLongitude(other.itsSubSatelliteLongitude),
      itsSubSatelliteLatitude(other.itsSubSatelliteLatitude),
      itsSweepAxis(other.itsSweepAxis),
      itsA(other.itsA),
      itsB(other.itsB),
      itsH(other.itsH),
      itsLon0(other.itsLon0),
      itsFalseEasting(other.itsFalseEasting),
      itsFalseNorthing(other.itsFalseNorthing)
{
	itsLogger = logger("geostationary_grid");
	itsSpatialReference = std::unique_ptr<OGRSpatialReference>(other.itsSpatialReference->Clone());
	itsLatLonToXYTransformer = std::unique_ptr<OGRCoordinateTransformation>(other.itsLatLonToXYTransformer->Clone());
	itsXYToLatLonTransformer = std::unique_ptr<OGRCoordinateTransformation>(other.itsXYToLatLonTransformer->Clone());
}

bool geostationary_grid::IsSupportedScanningMode(HPScanningMode mode)
{
	return mode == kTopLeft;
}

void geostationary_grid::Init(const earth_shape<double>& earthShape)
{
	if (!IsSupportedScanningMode(itsScanningMode))
	{
		itsLogger.Fatal(fmt::format("Unsupported scanning mode for geostationary grid: {}",
		                            HPScanningModeToString.at(itsScanningMode)));
		himan::Abort();
	}

	if (itsSweepAxis != 'x' && itsSweepAxis != 'y')
	{
		itsLogger.Fatal(fmt::format("Invalid sweep axis: '{}', must be 'x' or 'y'", itsSweepAxis));
		himan::Abort();
	}

	if (itsSubSatelliteLatitude != 0)
	{
		itsLogger.Fatal(fmt::format("Sub-satellite point latitude must be zero, got {}",
		                            static_cast<double>(itsSubSatelliteLatitude) * 1e-6));
		himan::Abort();
	}

	// Nr is in units of equatorial radius * 10^6. Missing value (all bits set) would
	// mean orthographic view, which is not supported.

	if (itsNr <= 1000000 || itsNr == 0xFFFFFFFF)
	{
		itsLogger.Fatal(fmt::format("Invalid or unsupported Nr: {}", itsNr));
		himan::Abort();
	}

	if (itsDx <= 0 || itsDy <= 0 || itsNi == 0 || itsNj == 0)
	{
		itsLogger.Fatal(fmt::format("Invalid grid size: ni={} nj={} dx={} dy={}", itsNi, itsNj, itsDx, itsDy));
		himan::Abort();
	}

	itsA = earthShape.A();
	itsB = earthShape.B();

	const double nr = static_cast<double>(itsNr) * 1e-6;
	itsH = (nr - 1.) * itsA;
	itsLon0 = NormalizeLongitude(static_cast<double>(itsSubSatelliteLongitude) * 1e-6);

	// Grid length in projection space (metres). Projection coordinates of geos
	// are scanning angle (radians) multiplied with satellite height.
	//
	// dx and dy are the apparent diameter of earth in grid lengths. Angular size of
	// earth is 2 * asin(1/Nr); in y direction it is scaled with the ratio of polar
	// and equatorial radius. This matches the ecCodes implementation, which is
	// used as the reference.

	const double angularSize = 2. * asin(1. / nr);

	itsDi = angularSize / static_cast<double>(itsDx) * itsH;
	itsDj = (itsB / itsA) * angularSize / static_cast<double>(itsDy) * itsH;

	// Himan regular_grid convention is that the first grid point is at projected
	// coordinates (0,0); set false easting and northing accordingly.

	itsFalseEasting = static_cast<double>(itsXp) * 1e-3 * itsDi;
	itsFalseNorthing = -static_cast<double>(itsYp) * 1e-3 * itsDj;

	const std::string ref = fmt::format(
	    "+proj=geos +h={:.6f} +lon_0={:.6f} +sweep={} +x_0={:.6f} +y_0={:.6f} +a={:.6f} +b={:.6f} "
	    "+units=m +no_defs +wktext",
	    itsH, itsLon0, itsSweepAxis, itsFalseEasting, itsFalseNorthing, itsA, itsB);

	itsSpatialReference = std::unique_ptr<OGRSpatialReference>(new OGRSpatialReference());

	if (itsSpatialReference->importFromProj4(ref.c_str()) != OGRERR_NONE)
	{
		itsLogger.Fatal(fmt::format("Failed to create spatial reference from '{}'", ref));
		himan::Abort();
	}

#if GDAL_VERSION_MAJOR > 1
	// HIMAN-326: Force traditional axis order because Himan internally expects x,y order
	itsSpatialReference->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
#endif

	auto geogCS = std::unique_ptr<OGRSpatialReference>(itsSpatialReference->CloneGeogCS());

	itsLatLonToXYTransformer = std::unique_ptr<OGRCoordinateTransformation>(
	    OGRCreateCoordinateTransformation(geogCS.get(), itsSpatialReference.get()));
	itsXYToLatLonTransformer = std::unique_ptr<OGRCoordinateTransformation>(
	    OGRCreateCoordinateTransformation(itsSpatialReference.get(), geogCS.get()));

	ASSERT(itsLatLonToXYTransformer);
	ASSERT(itsXYToLatLonTransformer);

	itsLogger.Trace(ref);
}

point geostationary_grid::GeosToLatLon(double x, double y) const
{
	// Inverse of geostationary satellite view projection for an ellipsoid,
	// following the PROJ implementation (geos.cpp, e_inverse).
	// x, y: projection coordinates in metres, origin at sub-satellite point.

	const double rg1 = itsH / itsA;  // radius_g_1
	const double rg = 1. + rg1;      // radius_g
	const double C = rg * rg - 1.;
	const double rp = itsB / itsA;  // radius_p
	const double rp2 = rp * rp;

	x /= itsA;
	y /= itsA;

	double Vx = -1.;
	double Vy, Vz;

	if (itsSweepAxis == 'x')
	{
		Vz = tan(y / rg1);
		Vy = tan(x / rg1) * hypot(1., Vz);
	}
	else
	{
		Vy = tan(x / rg1);
		Vz = tan(y / rg1) * hypot(1., Vy);
	}

	double a = Vz / rp;
	a = Vy * Vy + a * a + Vx * Vx;
	const double b = 2. * rg * Vx;
	const double det = b * b - 4. * a * C;

	if (det < 0.)
	{
		// Line of sight does not intersect earth
		return point(MissingDouble(), MissingDouble());
	}

	const double k = (-b - sqrt(det)) / (2. * a);

	Vx = rg + k * Vx;
	Vy *= k;
	Vz *= k;

	const double lambda = atan2(Vy, Vx);
	double phi = atan(Vz * cos(lambda) / Vx);
	phi = atan(tan(phi) / rp2);

	return point(NormalizeLongitude(lambda * constants::kRad + itsLon0), phi * constants::kRad);
}

point geostationary_grid::LatLonToGeos(const point& latlon) const
{
	// Forward geostationary satellite view projection for an ellipsoid,
	// following the PROJ implementation (geos.cpp, e_forward).

	if (IsMissing(latlon.X()) || IsMissing(latlon.Y()))
	{
		return point(MissingDouble(), MissingDouble());
	}

	const double rg1 = itsH / itsA;
	const double rg = 1. + rg1;
	const double rp = itsB / itsA;
	const double rp2 = rp * rp;

	const double lambda = NormalizeLongitude(latlon.X() - itsLon0) * constants::kDeg;
	double phi = latlon.Y() * constants::kDeg;

	// geodetic -> geocentric latitude
	phi = atan(rp2 * tan(phi));

	const double r = rp / hypot(rp * cos(phi), sin(phi));
	const double Vx = r * cos(lambda) * cos(phi);
	const double Vy = r * sin(lambda) * cos(phi);
	const double Vz = r * sin(phi);

	const double tmp = rg - Vx;

	// Check visibility
	if (tmp * Vx - Vy * Vy - Vz * Vz / rp2 < 0.)
	{
		return point(MissingDouble(), MissingDouble());
	}

	double x, y;

	if (itsSweepAxis == 'x')
	{
		x = rg1 * atan(Vy / hypot(Vz, tmp));
		y = rg1 * atan(Vz / tmp);
	}
	else
	{
		x = rg1 * atan(Vy / tmp);
		y = rg1 * atan(Vz / hypot(Vy, tmp));
	}

	return point(x * itsA, y * itsA);
}

point geostationary_grid::LatLon(size_t locationIndex) const
{
	ASSERT(locationIndex < itsNi * itsNj);

	const size_t j = locationIndex / itsNi;
	const size_t i = locationIndex % itsNi;

	// kTopLeft: i grows to east, j grows to south

	const double x = (static_cast<double>(i) - static_cast<double>(itsXp) * 1e-3) * itsDi;
	const double y = (static_cast<double>(itsYp) * 1e-3 - static_cast<double>(j)) * itsDj;

	return GeosToLatLon(x, y);
}

point geostationary_grid::LatLon(const point& projected) const
{
	return GeosToLatLon(projected.X() - itsFalseEasting, projected.Y() - itsFalseNorthing);
}

point geostationary_grid::Projected(const point& latlon) const
{
	const point g = LatLonToGeos(latlon);

	if (IsMissing(g.X()))
	{
		return g;
	}

	return point(g.X() + itsFalseEasting, g.Y() + itsFalseNorthing);
}

point geostationary_grid::ProjectedFirstPoint() const
{
	return point(-itsFalseEasting, -itsFalseNorthing);
}

std::vector<point> geostationary_grid::GridPointsInProjectionSpace() const
{
	std::vector<point> ret;
	ret.reserve(Size());

	for (size_t y = 0; y < itsNj; y++)
	{
		const double py = -static_cast<double>(y) * itsDj;

		for (size_t x = 0; x < itsNi; x++)
		{
			ret.emplace_back(static_cast<double>(x) * itsDi, py);
		}
	}

	return ret;
}

bool geostationary_grid::IsSameProjection(const geostationary_grid& other) const
{
	// Earth axes are compared with millimeter precision, as the values
	// can be decoded from grib with different floating point rounding

	return itsSweepAxis == other.itsSweepAxis && itsNr == other.itsNr &&
	       itsSubSatelliteLongitude == other.itsSubSatelliteLongitude &&
	       itsSubSatelliteLatitude == other.itsSubSatelliteLatitude &&
	       llround(itsA * 1000.) == llround(other.itsA * 1000.) && llround(itsB * 1000.) == llround(other.itsB * 1000.);
}

std::vector<point> geostationary_grid::XY(const regular_grid& target) const
{
	const double ni = static_cast<double>(itsNi - 1);
	const double nj = static_cast<double>(itsNj - 1);

	std::vector<point> sourceXY;
	sourceXY.reserve(target.Size());

	auto Append = [&](double x, double y)
	{
		if (!(x >= 0. && x <= ni && y >= 0. && y <= nj))
		{
			x = y = MissingDouble();
		}
		sourceXY.emplace_back(x, y);
	};

	const geostationary_grid* svtarget = dynamic_cast<const geostationary_grid*>(&target);

	if (svtarget && IsSameProjection(*svtarget))
	{
		// Same satellite projection: target and source grid differ only in
		// resolution and offset, so coordinates can be computed directly in
		// projection space without any transformations.

		itsLogger.Trace("Target is geostationary grid with same projection, no need to do transformation");

		const double txp = static_cast<double>(svtarget->itsXp) * 1e-3;
		const double typ = static_cast<double>(svtarget->itsYp) * 1e-3;
		const double xp = static_cast<double>(itsXp) * 1e-3;
		const double yp = static_cast<double>(itsYp) * 1e-3;

		for (size_t j = 0; j < svtarget->Nj(); j++)
		{
			const double gy = (typ - static_cast<double>(j)) * svtarget->Dj();
			const double y = yp - gy / itsDj;

			for (size_t i = 0; i < svtarget->Ni(); i++)
			{
				const double gx = (static_cast<double>(i) - txp) * svtarget->Di();
				Append(gx / itsDi + xp, y);
			}
		}

		return sourceXY;
	}

	// Generic case: transform target grid points from target projection space
	// to source projection space. Points that are not visible from the satellite
	// will fail to transform; mark them missing instead of aborting.

	const auto targetProj = target.GridPointsInProjectionSpace();

	std::vector<double> xs, ys;
	xs.reserve(targetProj.size());
	ys.reserve(targetProj.size());

	for (const auto& p : targetProj)
	{
		xs.push_back(p.X());
		ys.push_back(p.Y());
	}

	std::vector<int> success(targetProj.size(), 0);

	auto tosp = target.SpatialReference();
	auto xform = std::unique_ptr<OGRCoordinateTransformation>(
	    OGRCreateCoordinateTransformation(tosp.get(), itsSpatialReference.get()));

	if (!xform)
	{
		itsLogger.Fatal("Failed to create coordinate transformation");
		himan::Abort();
	}

	// Return value is false if any of the points failed; that is expected here
	xform->Transform(static_cast<int>(xs.size()), xs.data(), ys.data(), nullptr, success.data());

	for (size_t i = 0; i < xs.size(); i++)
	{
		if (!success[i] || !std::isfinite(xs[i]) || !std::isfinite(ys[i]))
		{
			sourceXY.emplace_back(MissingDouble(), MissingDouble());
			continue;
		}

		Append(xs[i] / itsDi, -ys[i] / itsDj);
	}

	return sourceXY;
}

size_t geostationary_grid::Hash() const
{
	std::vector<size_t> hashes;
	hashes.push_back(Type());
	hashes.push_back(Ni());
	hashes.push_back(Nj());
	hashes.push_back(ScanningMode());
	hashes.push_back(std::hash<long>{}(itsNr));
	hashes.push_back(std::hash<long>{}(itsDx));
	hashes.push_back(std::hash<long>{}(itsDy));
	hashes.push_back(std::hash<long>{}(itsXp));
	hashes.push_back(std::hash<long>{}(itsYp));
	hashes.push_back(std::hash<long>{}(itsSubSatelliteLongitude));
	hashes.push_back(std::hash<long>{}(itsSubSatelliteLatitude));
	hashes.push_back(std::hash<char>{}(itsSweepAxis));
	hashes.push_back(std::hash<long long>{}(llround(itsA * 1000.)));
	hashes.push_back(std::hash<long long>{}(llround(itsB * 1000.)));
	return boost::hash_range(hashes.begin(), hashes.end());
}

bool geostationary_grid::operator!=(const grid& other) const
{
	return !(other == *this);
}

bool geostationary_grid::operator==(const grid& other) const
{
	const geostationary_grid* g = dynamic_cast<const geostationary_grid*>(&other);

	if (g)
	{
		return EqualsTo(*g);
	}

	return false;
}

bool geostationary_grid::EqualsTo(const geostationary_grid& other) const
{
	// regular_grid::EqualsTo compares corner points in latlon, which are not
	// defined for full disk grids. Compare native parameters instead.

	if (!grid::EqualsTo(other))
	{
		return false;
	}

	if (itsNi != other.itsNi || itsNj != other.itsNj)
	{
		itsLogger.Trace(
		    fmt::format("Grid size does not match: {}x{} vs {}x{}", itsNi, itsNj, other.itsNi, other.itsNj));
		return false;
	}

	if (itsScanningMode != other.itsScanningMode)
	{
		itsLogger.Trace("Scanning mode does not match");
		return false;
	}

	if (itsDx != other.itsDx || itsDy != other.itsDy || itsXp != other.itsXp || itsYp != other.itsYp)
	{
		itsLogger.Trace(
		    fmt::format("Grid definition does not match: dx={} dy={} xp={} yp={} vs dx={} dy={} xp={} yp={}", itsDx,
		                itsDy, itsXp, itsYp, other.itsDx, other.itsDy, other.itsXp, other.itsYp));
		return false;
	}

	if (!IsSameProjection(other))
	{
		itsLogger.Trace("Projection does not match");
		itsLogger.Trace("This: " + Proj4String());
		itsLogger.Trace("Other: " + other.Proj4String());
		return false;
	}

	return true;
}

std::unique_ptr<grid> geostationary_grid::Clone() const
{
	return std::unique_ptr<grid>(new geostationary_grid(*this));
}

std::ostream& geostationary_grid::Write(std::ostream& file) const
{
	regular_grid::Write(file);

	file << "__itsNr__ " << itsNr << std::endl
	     << "__itsDx__ " << itsDx << std::endl
	     << "__itsDy__ " << itsDy << std::endl
	     << "__itsXp__ " << itsXp << std::endl
	     << "__itsYp__ " << itsYp << std::endl
	     << "__itsSubSatelliteLongitude__ " << itsSubSatelliteLongitude << std::endl
	     << "__itsSubSatelliteLatitude__ " << itsSubSatelliteLatitude << std::endl
	     << "__itsSweepAxis__ " << itsSweepAxis << std::endl;

	return file;
}

long geostationary_grid::Nr() const
{
	return itsNr;
}
long geostationary_grid::Dx() const
{
	return itsDx;
}
long geostationary_grid::Dy() const
{
	return itsDy;
}
long geostationary_grid::Xp() const
{
	return itsXp;
}
long geostationary_grid::Yp() const
{
	return itsYp;
}
long geostationary_grid::SubSatelliteLongitude() const
{
	return itsSubSatelliteLongitude;
}
long geostationary_grid::SubSatelliteLatitude() const
{
	return itsSubSatelliteLatitude;
}
char geostationary_grid::SweepAxis() const
{
	return itsSweepAxis;
}
double geostationary_grid::Height() const
{
	return itsH;
}
double geostationary_grid::Orientation() const
{
	return itsLon0;
}
