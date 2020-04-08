#include "lambert_equal_area_grid.h"
#include "info.h"
#include <cpl_conv.h>
#include <functional>
#include <ogr_spatialref.h>
using namespace himan;
using namespace himan::plugin;

lambert_equal_area_grid::lambert_equal_area_grid(HPScanningMode theScanningMode, const point& theFirstPoint, size_t ni,
                                                 size_t nj, double di, double dj,
                                                 std::unique_ptr<OGRSpatialReference> spRef, bool firstPointIsProjected)
    : regular_grid(kLambertEqualArea, theScanningMode, false), itsDi(di), itsDj(dj), itsNi(ni), itsNj(nj)
{
	itsLogger = logger("lambert_equal_area_grid");
	itsSpatialReference = std::move(spRef);
	CreateCoordinateTransformations(theFirstPoint, firstPointIsProjected);
}

lambert_equal_area_grid::lambert_equal_area_grid(HPScanningMode theScanningMode, const point& theFirstPoint, size_t ni,
                                                 size_t nj, double di, double dj, double theOrientation,
                                                 double theStandardParallel, const earth_shape<double>& earthShape,
                                                 bool firstPointIsProjected)
    : regular_grid(kLambertEqualArea, theScanningMode, false), itsDi(di), itsDj(dj), itsNi(ni), itsNj(nj)
{
	itsLogger = logger("lambert_equal_area_grid");

	std::stringstream ss;
	ss << "+proj=laea +lat_0=" << theStandardParallel << " +lon_0=" << theOrientation << " +a=" << std::fixed
	   << earthShape.A() << " +b=" << earthShape.B() << " +units=m +no_defs +wktext";

	itsSpatialReference = std::unique_ptr<OGRSpatialReference>(new OGRSpatialReference());
	itsSpatialReference->importFromProj4(ss.str().c_str());

	CreateCoordinateTransformations(theFirstPoint, firstPointIsProjected);
}

lambert_equal_area_grid::lambert_equal_area_grid(const lambert_equal_area_grid& other)
    : regular_grid(other), itsDi(other.itsDi), itsDj(other.itsDj), itsNi(other.itsNi), itsNj(other.itsNj)
{
	itsLogger = logger("lambert_equal_area_grid");
	itsSpatialReference = std::unique_ptr<OGRSpatialReference>(other.itsSpatialReference->Clone());

	CreateCoordinateTransformations(other.FirstPoint(), false);
}

lambert_equal_area_grid::~lambert_equal_area_grid() = default;

void lambert_equal_area_grid::CreateCoordinateTransformations(const point& firstPoint, bool firstPointIsProjected)
{
	ASSERT(itsSpatialReference);
	ASSERT(itsSpatialReference->IsProjected());

	auto geogCS = std::unique_ptr<OGRSpatialReference>(itsSpatialReference->CloneGeogCS());

	itsLatLonToXYTransformer = std::unique_ptr<OGRCoordinateTransformation>(
	    OGRCreateCoordinateTransformation(geogCS.get(), itsSpatialReference.get()));
	itsXYToLatLonTransformer = std::unique_ptr<OGRCoordinateTransformation>(
	    OGRCreateCoordinateTransformation(itsSpatialReference.get(), geogCS.get()));

	double lat = firstPoint.Y(), lon = firstPoint.X();

	if (firstPointIsProjected)
	{
		if (!itsXYToLatLonTransformer->Transform(1, &lon, &lat))
		{
			itsLogger.Error("Failed to get first point latlon");
			return;
		}
	}

	if (!itsLatLonToXYTransformer->Transform(1, &lon, &lat))
	{
		itsLogger.Error("Failed to get false easting and northing");
		return;
	}

	if (fabs(lon) < 1e-4 and fabs(lat) < 1e-4)
	{
		return;
	}

	const double orientation = itsSpatialReference->GetProjParm(SRS_PP_CENTRAL_MERIDIAN, 0.0);
	const double parallel = itsSpatialReference->GetProjParm(SRS_PP_LATITUDE_OF_CENTER, 0.0);
	const double fe = itsSpatialReference->GetProjParm(SRS_PP_FALSE_EASTING, 0.0) - lon;
	const double fn = itsSpatialReference->GetProjParm(SRS_PP_FALSE_NORTHING, 0.0) - lat;

	if (itsSpatialReference->SetLAEA(parallel, orientation, fe, fn) != OGRERR_NONE)
	{
		itsLogger.Error("Failed to create LAEA projection");
		return;
	}

	itsLatLonToXYTransformer = std::unique_ptr<OGRCoordinateTransformation>(
	    OGRCreateCoordinateTransformation(geogCS.get(), itsSpatialReference.get()));
	itsXYToLatLonTransformer = std::unique_ptr<OGRCoordinateTransformation>(
	    OGRCreateCoordinateTransformation(itsSpatialReference.get(), geogCS.get()));

	char* projstr;
	if (itsSpatialReference->exportToProj4(&projstr) != OGRERR_NONE)
	{
		throw std::runtime_error("Failed to get proj4 str");
	}

	itsLogger.Trace(projstr);
	CPLFree(projstr);
}

size_t lambert_equal_area_grid::Size() const
{
	if (itsNi == static_cast<size_t>(kHPMissingInt) || itsNj == static_cast<size_t>(kHPMissingInt))
	{
		return kHPMissingInt;
	}

	return itsNi * itsNj;
}

point lambert_equal_area_grid::TopRight() const
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
			throw std::runtime_error("Unhandled scanning mode: " + HPScanningModeToString.at(itsScanningMode));
	}
}

point lambert_equal_area_grid::BottomLeft() const
{
	switch (itsScanningMode)
	{
		case kBottomLeft:
			return LatLon(0);
		case kTopLeft:
			if (itsNi == static_cast<size_t>(kHPMissingInt) || itsNj == static_cast<size_t>(kHPMissingInt))
			{
				return point();
			}

			return LatLon(itsNj * itsNi - itsNi);
		case kUnknownScanningMode:
			return point();
		default:
			throw std::runtime_error("Unhandled scanning mode: " + HPScanningModeToString.at(itsScanningMode));
	}
}

point lambert_equal_area_grid::TopLeft() const
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
			return LatLon(0);
		case kUnknownScanningMode:
			return point();
		default:
			throw std::runtime_error("Unhandled scanning mode: " + HPScanningModeToString.at(itsScanningMode));
	}
}

point lambert_equal_area_grid::BottomRight() const
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
			throw std::runtime_error("Unhandled scanning mode: " + HPScanningModeToString.at(itsScanningMode));
	}
}

point lambert_equal_area_grid::FirstPoint() const
{
	return LatLon(0);
}

point lambert_equal_area_grid::LastPoint() const
{
	return LatLon(itsNi * itsNj - 1);
}

point lambert_equal_area_grid::XY(const point& latlon) const
{
	double projX = latlon.X(), projY = latlon.Y();
	ASSERT(itsLatLonToXYTransformer);

	// 1. Transform latlon to projected coordinates.
	// Projected coordinates are in meters, with false easting and
	// false northing applied so that point 0,0 is top left or bottom left,
	// depending on the scanning mode.

	if (!itsLatLonToXYTransformer->Transform(1, &projX, &projY))
	{
		itsLogger.Error("Error determining xy value for latlon point " + std::to_string(latlon.X()) + "," +
		                std::to_string(latlon.Y()));
		return point();
	}

	// 2. Transform projected coordinates (meters) to grid xy (no unit).
	// Projected coordinates run from 0 ... area width and 0 ... area height.
	// Grid point coordinates run from 0 ... ni and 0 ... nj.

	const double x = (projX / itsDi);
	const double y = (projY / itsDj) * (itsScanningMode == kTopLeft ? -1 : 1);

	if (x < 0 || x > Ni() - 1 || y < 0 || y > Nj() - 1)
	{
		return point(MissingDouble(), MissingDouble());
	}

	return point(x, y);
}

point lambert_equal_area_grid::LatLon(size_t locationIndex) const
{
	ASSERT(itsNi != static_cast<size_t>(kHPMissingInt));
	ASSERT(itsNj != static_cast<size_t>(kHPMissingInt));
	ASSERT(!IsKHPMissingValue(Di()));
	ASSERT(!IsKHPMissingValue(Dj()));
	ASSERT(locationIndex < itsNi * itsNj);

	const size_t jIndex = static_cast<size_t>(locationIndex / itsNi);
	const size_t iIndex = static_cast<size_t>(locationIndex % itsNi);

	double x = static_cast<double>(iIndex) * Di();
	double y = static_cast<double>(jIndex) * Dj() * (itsScanningMode == kTopLeft ? -1 : 1);

	ASSERT(itsXYToLatLonTransformer);
	if (!itsXYToLatLonTransformer->Transform(1, &x, &y))
	{
		itsLogger.Error("Error determining latitude longitude value for xy point " + std::to_string(x) + "," +
		                std::to_string(y));
		return point();
	}

	return point(x, y);
}

size_t lambert_equal_area_grid::Hash() const
{
	std::vector<size_t> hashes;
	hashes.push_back(Type());
	hashes.push_back(FirstPoint().Hash());
	hashes.push_back(Ni());
	hashes.push_back(Nj());
	hashes.push_back(std::hash<double>{}(Di()));
	hashes.push_back(std::hash<double>{}(Dj()));
	hashes.push_back(ScanningMode());
	hashes.push_back(std::hash<double>{}(Orientation()));
	hashes.push_back(std::hash<double>{}(StandardParallel()));
	return boost::hash_range(hashes.begin(), hashes.end());
}

void lambert_equal_area_grid::Ni(size_t theNi)
{
	itsNi = theNi;
}
void lambert_equal_area_grid::Nj(size_t theNj)
{
	itsNj = theNj;
}
size_t lambert_equal_area_grid::Ni() const
{
	return itsNi;
}
size_t lambert_equal_area_grid::Nj() const
{
	return itsNj;
}
void lambert_equal_area_grid::Di(double theDi)
{
	itsDi = theDi;
}
void lambert_equal_area_grid::Dj(double theDj)
{
	itsDj = theDj;
}
double lambert_equal_area_grid::Di() const
{
	return itsDi;
}
double lambert_equal_area_grid::Dj() const
{
	return itsDj;
}
bool lambert_equal_area_grid::operator!=(const grid& other) const
{
	return !(other == *this);
}
bool lambert_equal_area_grid::operator==(const grid& other) const
{
	const lambert_equal_area_grid* g = dynamic_cast<const lambert_equal_area_grid*>(&other);

	if (g)
	{
		return EqualsTo(*g);
	}

	return false;
}

bool lambert_equal_area_grid::EqualsTo(const lambert_equal_area_grid& other) const
{
	if (!regular_grid::EqualsTo(other))
	{
		return false;
	}

	if (BottomLeft() != other.BottomLeft())
	{
		itsLogger.Trace("BottomLeft does not match: X " + std::to_string(BottomLeft().X()) + " vs " +
		                std::to_string(other.BottomLeft().X()));
		itsLogger.Trace("BottomLeft does not match: Y " + std::to_string(BottomLeft().Y()) + " vs " +
		                std::to_string(other.BottomLeft().Y()));
		return false;
	}

	if (TopLeft() != other.TopLeft())
	{
		itsLogger.Trace("TopLeft does not match: X " + std::to_string(TopLeft().X()) + " vs " +
		                std::to_string(other.TopLeft().X()));
		itsLogger.Trace("TopLeft does not match: Y " + std::to_string(TopLeft().Y()) + " vs " +
		                std::to_string(other.TopLeft().Y()));
		return false;
	}

	const double kEpsilon = 0.0001;

	if (fabs(itsDi - other.itsDi) > kEpsilon)
	{
		itsLogger.Trace("Di does not match: " + std::to_string(Di()) + " vs " + std::to_string(other.Di()));
		return false;
	}

	if (fabs(itsDj - other.itsDj) > kEpsilon)
	{
		itsLogger.Trace("Dj does not match: " + std::to_string(Dj()) + " vs " + std::to_string(other.Dj()));
		return false;
	}

	if (itsNi != other.Ni())
	{
		itsLogger.Trace("Ni does not match: " + std::to_string(itsNi) + " vs " + std::to_string(other.Ni()));
		return false;
	}

	if (itsNj != other.Nj())
	{
		itsLogger.Trace("Nj does not match: " + std::to_string(itsNj) + " vs " + std::to_string(other.Nj()));
		return false;
	}

	if (Orientation() != other.Orientation())
	{
		itsLogger.Trace("Orientation does not match: " + std::to_string(Orientation()) + " vs " +
		                std::to_string(other.Orientation()));
		return false;
	}

	if (StandardParallel() != other.StandardParallel())
	{
		itsLogger.Trace("Standard latitude does not match: " + std::to_string(StandardParallel()) + " vs " +
		                std::to_string(other.StandardParallel()));
		return false;
	}

	return true;
}

std::unique_ptr<grid> lambert_equal_area_grid::Clone() const
{
	return std::unique_ptr<grid>(new lambert_equal_area_grid(*this));
}

std::ostream& lambert_equal_area_grid::Write(std::ostream& file) const
{
	regular_grid::Write(file);

	file << "__itsNi__ " << itsNi << std::endl
	     << "__itsNj__ " << itsNj << std::endl
	     << "__itsDi__ " << Di() << std::endl
	     << "__itsDj__ " << Dj() << std::endl
	     << "__proj4String__ " << Proj4String() << std::endl;

	return file;
}

double lambert_equal_area_grid::Orientation() const
{
	return itsSpatialReference->GetProjParm(SRS_PP_CENTRAL_MERIDIAN, MissingDouble());
}

double lambert_equal_area_grid::StandardParallel() const
{
	return itsSpatialReference->GetProjParm(SRS_PP_LATITUDE_OF_CENTER, MissingDouble());
}

OGRSpatialReference lambert_equal_area_grid::SpatialReference() const
{
	return OGRSpatialReference(*itsSpatialReference);
}
