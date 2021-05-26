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
    : regular_grid(kStereographic, theScanningMode, di, dj, ni, nj, false, theName)
{
	itsLogger = logger("stereographic_grid");

	if (IsMissing(earthShape.A()) && IsMissing(earthShape.B()))
	{
		itsLogger.Fatal("Cannot create area without knowing earth shape");
		himan::Abort();
	}

	const std::string ref = fmt::format("+proj=stere +lat_0=90 +lat_ts=60 +lon_0={} +k=1 +units=m {} +wktext +no_defs",
	                                    theOrientation, earthShape.Proj4String());

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
	const double fe = itsSpatialReference->GetProjParm(SRS_PP_FALSE_EASTING, 0.0) - lon;
	const double fn = itsSpatialReference->GetProjParm(SRS_PP_FALSE_NORTHING, 0.0) - lat;
	const auto es = EarthShape();

	// SetStereographic() has no argument for lat_0
	// Have to build proj4 string

	std::stringstream ss;

	ss << "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=" << orientation << " +k=1 +units=m"
	   << " +a=" << fixed << es.A() << " +b=" << es.B() << " +wktext +no_defs"
	   << " +x_0=" << fe << " +y_0=" << fn;

	itsSpatialReference->importFromProj4(ss.str().c_str());

	itsLatLonToXYTransformer = std::unique_ptr<OGRCoordinateTransformation>(
	    OGRCreateCoordinateTransformation(geogCS.get(), itsSpatialReference.get()));
	itsXYToLatLonTransformer = std::unique_ptr<OGRCoordinateTransformation>(
	    OGRCreateCoordinateTransformation(itsSpatialReference.get(), geogCS.get()));
}

double stereographic_grid::Orientation() const
{
	return itsSpatialReference->GetProjParm(SRS_PP_CENTRAL_MERIDIAN, MissingDouble());
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
