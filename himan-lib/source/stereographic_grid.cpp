#include "stereographic_grid.h"
#include "util.h"
#include <functional>
#include <ogr_spatialref.h>

using namespace himan;
using namespace std;

stereographic_grid::stereographic_grid(HPScanningMode theScanningMode, const point& theFirstPoint, size_t ni, size_t nj,
                                       double di, double dj, double theOrientation,
                                       const earth_shape<double>& earthShape, bool firstPointIsProjected,
                                       const std::string& theName)
    : stereographic_grid(theScanningMode, theFirstPoint, ni, nj, di, dj, theOrientation, 90., 60., earthShape,
                         firstPointIsProjected, theName)
{
}

stereographic_grid::stereographic_grid(HPScanningMode theScanningMode, const point& theFirstPoint, size_t ni, size_t nj,
                                       double di, double dj, double theOrientation, double latin, double lat_ts,
                                       const earth_shape<double>& earthShape, bool firstPointIsProjected,
                                       const std::string& theName)
    : regular_grid(kStereographic, theScanningMode, di, dj, ni, nj, false, theName)
{
	itsLogger = logger("stereographic_grid");

	if (IsMissing(earthShape.A()) && IsMissing(earthShape.B()))
	{
		itsLogger.Fatal("Cannot create area without knowing earth shape");
		himan::Abort();
	}

	const std::string ref = fmt::format("+proj=stere +lat_0={} +lat_ts={} +lon_0={} +k=1 +units=m {} +wktext +no_defs",
	                                    latin, lat_ts, theOrientation, earthShape.Proj4String());
	itsSpatialReference = std::unique_ptr<OGRSpatialReference>(new OGRSpatialReference());
	itsSpatialReference->importFromProj4(ref.c_str());

#if GDAL_VERSION_MAJOR > 1
	itsSpatialReference->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
#endif

	CreateCoordinateTransformations(theFirstPoint, firstPointIsProjected);
	itsLogger.Trace(Proj4String());
}

stereographic_grid::stereographic_grid(HPScanningMode theScanningMode, const point& theFirstPoint, size_t ni, size_t nj,
                                       double di, double dj, std::unique_ptr<OGRSpatialReference> spRef,
                                       bool firstPointIsProjected, const std::string& theName)
    : regular_grid(kStereographic, theScanningMode, di, dj, ni, nj, false, theName)
{
	itsLogger = logger("stereographic_grid");
	itsSpatialReference = std::move(spRef);

#if GDAL_VERSION_MAJOR > 1
	itsSpatialReference->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
#endif

	CreateCoordinateTransformations(theFirstPoint, firstPointIsProjected);
	itsLogger.Trace(Proj4String());
}

stereographic_grid::stereographic_grid(const stereographic_grid& other) : regular_grid(other)
{
	itsLogger = logger("stereographic_grid");
	itsSpatialReference = std::unique_ptr<OGRSpatialReference>(other.itsSpatialReference->Clone());
	CreateCoordinateTransformations(other.FirstPoint(), false);
}

void stereographic_grid::CreateCoordinateTransformations(const point& firstPoint, bool firstPointIsProjected)
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
	const double latin = LatitudeOfCenter();
	const double fe = itsSpatialReference->GetProjParm(SRS_PP_FALSE_EASTING, 0.0) - lon;
	const double fn = itsSpatialReference->GetProjParm(SRS_PP_FALSE_NORTHING, 0.0) - lat;
	const auto es = EarthShape();

	// SetStereographic() has no argument for lat_0
	// Have to build proj4 string

	std::string ref =
	    fmt::format("+proj=stere +lat_0={} +lon_0={} +k=1 +units=m +a={} +b={} +wktext +no_defs +x_0={} +y_0={}", latin,
	                orientation, es.A(), es.B(), fe, fn);

	if (latin == 90. || latin == -90.)
	{
		// only taken into account for Polar Stereographic formulations (+lat_0 = +/- 90 ), and then defaults to the
		// +lat_0 value
		double lat_ts = LatitudeOfOrigin();
		if (IsMissing(lat_ts))
		{
			lat_ts = latin;
		}

		ref = fmt::format("{} +lat_ts={}", ref, lat_ts);
	}

	itsSpatialReference->importFromProj4(ref.c_str());

	itsLatLonToXYTransformer = std::unique_ptr<OGRCoordinateTransformation>(
	    OGRCreateCoordinateTransformation(geogCS.get(), itsSpatialReference.get()));
	itsXYToLatLonTransformer = std::unique_ptr<OGRCoordinateTransformation>(
	    OGRCreateCoordinateTransformation(itsSpatialReference.get(), geogCS.get()));
}

double stereographic_grid::Orientation() const
{
	return itsSpatialReference->GetProjParm(SRS_PP_CENTRAL_MERIDIAN, MissingDouble());
}

double stereographic_grid::LatitudeOfCenter() const
{
	// kludge-ish: gdal seems to not store 'lat_0' anywhere, although it should be in
	// SRS_PP_LATITUDE_OF_CENTER
	//
	// check if this is POLAR stereographic and assing value 90/-90

	double latitudeOfCenter = itsSpatialReference->GetProjParm(SRS_PP_LATITUDE_OF_CENTER, MissingDouble());

	if (IsMissing(latitudeOfCenter))
	{
		const std::string proj = std::string(itsSpatialReference->GetAttrValue("PROJECTION"));

		if (proj == SRS_PT_POLAR_STEREOGRAPHIC)
		{
			latitudeOfCenter = (LatitudeOfOrigin() > 0) ? 90 : -90;
		}
	}

	return latitudeOfCenter;
}

double stereographic_grid::LatitudeOfOrigin() const
{
	return itsSpatialReference->GetProjParm(SRS_PP_LATITUDE_OF_ORIGIN, MissingDouble());
}

size_t stereographic_grid::Hash() const
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
	hashes.push_back(hash<double>{}(LatitudeOfCenter()));
	hashes.push_back(hash<double>{}(LatitudeOfOrigin()));
	return boost::hash_range(hashes.begin(), hashes.end());
}

unique_ptr<grid> stereographic_grid::Clone() const
{
	return unique_ptr<grid>(new stereographic_grid(*this));
}

ostream& stereographic_grid::Write(std::ostream& file) const
{
	regular_grid::Write(file);
	file << "__itsOrientation__ " << Orientation() << endl;
	file << "__itsLatitudeOfCenter__ " << LatitudeOfCenter() << endl;
	file << "__itsLatitudeOfOrigin__ " << LatitudeOfOrigin() << endl;

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
