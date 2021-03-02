#include "transverse_mercator_grid.h"
#include "info.h"
#include <functional>
#include <ogr_spatialref.h>

using namespace himan;
using namespace himan::plugin;

transverse_mercator_grid::transverse_mercator_grid(HPScanningMode theScanningMode, const point& theFirstPoint,
                                                   size_t ni, size_t nj, double di, double dj,
                                                   std::unique_ptr<OGRSpatialReference> spRef,
                                                   bool firstPointIsProjected)
    : regular_grid(kTransverseMercator, theScanningMode, di, dj, ni, nj)
{
	itsLogger = logger("transverse_mercator_grid");
	itsSpatialReference = std::move(spRef);

#if GDAL_VERSION_MAJOR > 1
	// HIMAN-326: Force traditional axis order because Himan internally expects x,y order
	itsSpatialReference->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
#endif

	CreateCoordinateTransformations(theFirstPoint, firstPointIsProjected);
	itsLogger.Trace(Proj4String());
}

transverse_mercator_grid::transverse_mercator_grid(HPScanningMode scanningMode, const point& firstPoint, size_t ni,
                                                   size_t nj, double di, double dj, double orientation,
                                                   double standardParallel, double scale, double falseEasting,
                                                   double falseNorthing, const earth_shape<double>& earthShape,
                                                   bool firstPointIsProjected)
    : regular_grid(kTransverseMercator, scanningMode, di, dj, ni, nj)
{
	itsLogger = logger("transverse_mercator_grid");

	std::stringstream ss;
	ss << "+proj=tmerc +lat_0=" << standardParallel << " +lon_0=" << orientation << " +k_0=" << scale << std::fixed
	   << " +x_0=" << falseEasting << " +y_0=" << falseNorthing << " +a=" << earthShape.A() << " +b=" << earthShape.B()
	   << " +units=m +no_defs +wktext";

	itsSpatialReference = std::unique_ptr<OGRSpatialReference>(new OGRSpatialReference());
	itsSpatialReference->importFromProj4(ss.str().c_str());

#if GDAL_VERSION_MAJOR > 1
	itsSpatialReference->SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER);
#endif

	CreateCoordinateTransformations(firstPoint, firstPointIsProjected);
	itsLogger.Trace(Proj4String());
}

transverse_mercator_grid::transverse_mercator_grid(const transverse_mercator_grid& other) : regular_grid(other)
{
	itsLogger = logger("transverse_mercator_grid");
	itsSpatialReference = std::unique_ptr<OGRSpatialReference>(other.itsSpatialReference->Clone());

	CreateCoordinateTransformations(other.FirstPoint(), false);
}

void transverse_mercator_grid::CreateCoordinateTransformations(const point& firstPoint, bool firstPointIsProjected)
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

	if (fabs(lon) < 1e-4 and fabs(lat) < 1e-4)
	{
		return;
	}

	const double orientation = Orientation();
	const double parallel = StandardParallel();
	const double scale = Scale();
	const double fe = itsSpatialReference->GetProjParm(SRS_PP_FALSE_EASTING, 0.0) - lon;
	const double fn = itsSpatialReference->GetProjParm(SRS_PP_FALSE_NORTHING, 0.0) - lat;

	itsSpatialReference = std::unique_ptr<OGRSpatialReference>(itsSpatialReference->CloneGeogCS());
	if (itsSpatialReference->SetTM(parallel, orientation, scale, fe, fn) != OGRERR_NONE)
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

size_t transverse_mercator_grid::Hash() const
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
	hashes.push_back(std::hash<double>{}(Scale()));
	return boost::hash_range(hashes.begin(), hashes.end());
}

bool transverse_mercator_grid::operator!=(const grid& other) const
{
	return !(other == *this);
}
bool transverse_mercator_grid::operator==(const grid& other) const
{
	const transverse_mercator_grid* g = dynamic_cast<const transverse_mercator_grid*>(&other);

	if (g)
	{
		return EqualsTo(*g);
	}

	return false;
}

bool transverse_mercator_grid::EqualsTo(const transverse_mercator_grid& other) const
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

std::unique_ptr<grid> transverse_mercator_grid::Clone() const
{
	return std::unique_ptr<grid>(new transverse_mercator_grid(*this));
}

std::ostream& transverse_mercator_grid::Write(std::ostream& file) const
{
	regular_grid::Write(file);

	file << "__itsOrientation__ " << Orientation() << std::endl
	     << "__itsStandardParallel__ " << StandardParallel() << std::endl
	     << "__itsScale__" << Scale() << std::endl;

	return file;
}

double transverse_mercator_grid::Orientation() const
{
	return itsSpatialReference->GetProjParm(SRS_PP_CENTRAL_MERIDIAN, MissingDouble());
}

double transverse_mercator_grid::StandardParallel() const
{
	return itsSpatialReference->GetProjParm(SRS_PP_LATITUDE_OF_ORIGIN, MissingDouble());
}

double transverse_mercator_grid::Scale() const
{
	return itsSpatialReference->GetProjParm(SRS_PP_SCALE_FACTOR, MissingDouble());
}
