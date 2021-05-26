#include "lambert_equal_area_grid.h"
#include "info.h"
#include "util.h"
#include <functional>
#include <ogr_spatialref.h>

using namespace himan;
using namespace himan::plugin;

lambert_equal_area_grid::lambert_equal_area_grid(HPScanningMode theScanningMode, const point& theFirstPoint, size_t ni,
                                                 size_t nj, double di, double dj,
                                                 std::unique_ptr<OGRSpatialReference> spRef, bool firstPointIsProjected,
                                                 const std::string& theName)
    : regular_grid(kLambertEqualArea, theScanningMode, di, dj, ni, nj, false, theName)
{
	itsLogger = logger("lambert_equal_area_grid");
	itsSpatialReference = std::move(spRef);

#if GDAL_VERSION_MAJOR > 1
	itsSpatialReference->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
#endif

	CreateCoordinateTransformations(theFirstPoint, firstPointIsProjected);
	itsLogger.Trace(Proj4String());
}

lambert_equal_area_grid::lambert_equal_area_grid(HPScanningMode theScanningMode, const point& theFirstPoint, size_t ni,
                                                 size_t nj, double di, double dj, double theOrientation,
                                                 double theStandardParallel, const earth_shape<double>& earthShape,
                                                 bool firstPointIsProjected, const std::string& theName)
    : regular_grid(kLambertEqualArea, theScanningMode, di, dj, ni, nj, false, theName)
{
	itsLogger = logger("lambert_equal_area_grid");

	const std::string ref = fmt::format("+proj=laea +lat_0={} +lon_0={} {} +units=m +no_defs +wktext",
	                                    theStandardParallel, theOrientation, earthShape.Proj4String());

	itsSpatialReference = std::unique_ptr<OGRSpatialReference>(new OGRSpatialReference());
	itsSpatialReference->importFromProj4(ref.c_str());

#if GDAL_VERSION_MAJOR > 1
	itsSpatialReference->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
#endif

	CreateCoordinateTransformations(theFirstPoint, firstPointIsProjected);
	itsLogger.Trace(Proj4String());
}

lambert_equal_area_grid::lambert_equal_area_grid(const lambert_equal_area_grid& other) : regular_grid(other)
{
	itsLogger = logger("lambert_equal_area_grid");
	itsSpatialReference = std::unique_ptr<OGRSpatialReference>(other.itsSpatialReference->Clone());

	CreateCoordinateTransformations(other.FirstPoint(), false);
}

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

	if (firstPointIsProjected == false)
	{
		if (!itsLatLonToXYTransformer->Transform(1, &lon, &lat))
		{
			itsLogger.Fatal("Failed to get false easting and northing");
			himan::Abort();
		}
	}

	lon = util::round(lon, 4);
	lat = util::round(lat, 4);

	if (fabs(lon) < 1e-4 and fabs(lat) < 1e-4)
	{
		return;
	}

	const double orientation = Orientation();
	const double parallel = StandardParallel();
	const double fe = itsSpatialReference->GetProjParm(SRS_PP_FALSE_EASTING, 0.0) - lon;
	const double fn = itsSpatialReference->GetProjParm(SRS_PP_FALSE_NORTHING, 0.0) - lat;

	itsSpatialReference = std::unique_ptr<OGRSpatialReference>(itsSpatialReference->CloneGeogCS());
	if (itsSpatialReference->SetLAEA(parallel, orientation, fe, fn) != OGRERR_NONE)
	{
		itsLogger.Fatal("Failed to create projection");
		himan::Abort();
	}

	itsLatLonToXYTransformer = std::unique_ptr<OGRCoordinateTransformation>(
	    OGRCreateCoordinateTransformation(geogCS.get(), itsSpatialReference.get()));
	itsXYToLatLonTransformer = std::unique_ptr<OGRCoordinateTransformation>(
	    OGRCreateCoordinateTransformation(itsSpatialReference.get(), geogCS.get()));
	ASSERT(itsLatLonToXYTransformer);
	ASSERT(itsXYToLatLonTransformer);
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

	if (!itsSpatialReference->IsSame(other.itsSpatialReference.get()))
	{
		itsLogger.Trace("Areas are not equal");
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

	file << "__itsOrientation__ " << Orientation() << std::endl
	     << "__itsStandardParallel__ " << StandardParallel() << std::endl;

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
