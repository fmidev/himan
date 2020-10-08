#include "lambert_conformal_grid.h"
#include "info.h"
#include <functional>
#include <ogr_spatialref.h>

using namespace himan;
using namespace std;

lambert_conformal_grid::lambert_conformal_grid(HPScanningMode theScanningMode, const point& theFirstPoint, size_t ni,
                                               size_t nj, double di, double dj, double orientation,
                                               double standardParallel1, double standardParallel2,
                                               const earth_shape<double>& earthShape, bool firstPointIsProjected)
    : regular_grid(kLambertConformalConic, theScanningMode, di, dj, ni, nj)
{
	itsLogger = logger("lambert_conformal_grid");

	if (IsMissing(earthShape.A()) && IsMissing(earthShape.B()))
	{
		itsLogger.Fatal("Cannot create area without knowing earth shape");
		himan::Abort();
	}

	std::stringstream ss;

	ss << "+proj=lcc +lat_1=" << standardParallel1 << " +lat_2=" << standardParallel2 << " +lat_0=" << standardParallel1
	   << " +lon_0=" << orientation << " +a=" << fixed << earthShape.A() << " +b=" << earthShape.B()
	   << " +units=m +no_defs +wktext";

	itsSpatialReference = std::unique_ptr<OGRSpatialReference>(new OGRSpatialReference());
	itsSpatialReference->importFromProj4(ss.str().c_str());

	CreateCoordinateTransformations(theFirstPoint, firstPointIsProjected);
	itsLogger.Trace(Proj4String());
}

lambert_conformal_grid::lambert_conformal_grid(HPScanningMode theScanningMode, const point& theFirstPoint, size_t ni,
                                               size_t nj, double di, double dj,
                                               std::unique_ptr<OGRSpatialReference> spRef, bool firstPointIsProjected)
    : regular_grid(kLambertConformalConic, theScanningMode, di, dj, ni, nj)
{
	itsLogger = logger("lambert_conformal_grid");
	itsSpatialReference = std::move(spRef);
	CreateCoordinateTransformations(theFirstPoint, firstPointIsProjected);
	itsLogger.Trace(Proj4String());
}

lambert_conformal_grid::lambert_conformal_grid(const lambert_conformal_grid& other) : regular_grid(other)
{
	itsLogger = logger("lambert_conformal_grid");
	itsSpatialReference = std::unique_ptr<OGRSpatialReference>(other.itsSpatialReference->Clone());
	CreateCoordinateTransformations(other.FirstPoint(), false);
}

void lambert_conformal_grid::CreateCoordinateTransformations(const point& firstPoint, bool firstPointIsProjected)
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

	const double orientation = Orientation();
	const double parallel1 = StandardParallel1();
	const double parallel2 = StandardParallel2();

	const double fe = itsSpatialReference->GetProjParm(SRS_PP_FALSE_EASTING, 0.0) - lon;
	const double fn = itsSpatialReference->GetProjParm(SRS_PP_FALSE_NORTHING, 0.0) - lat;

	// need to recreate spatial reference to get SetLCC to accept new falsings
	itsSpatialReference = std::unique_ptr<OGRSpatialReference>(itsSpatialReference->CloneGeogCS());
	if (itsSpatialReference->SetLCC(parallel1, parallel2, parallel1, orientation, fe, fn) != OGRERR_NONE)
	{
		itsLogger.Error("Failed to create LCC projection");
		return;
	}

	itsLatLonToXYTransformer = std::unique_ptr<OGRCoordinateTransformation>(
	    OGRCreateCoordinateTransformation(geogCS.get(), itsSpatialReference.get()));
	itsXYToLatLonTransformer = std::unique_ptr<OGRCoordinateTransformation>(
	    OGRCreateCoordinateTransformation(itsSpatialReference.get(), geogCS.get()));
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

double lambert_conformal_grid::Orientation() const
{
	return itsSpatialReference->GetProjParm(SRS_PP_CENTRAL_MERIDIAN, MissingDouble());
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

	if (Orientation() != other.Orientation())
	{
		itsLogger.Trace("Orientation does not match: " + to_string(Orientation()) + " vs " +
		                to_string(other.Orientation()));
		return false;
	}

	if (StandardParallel1() != StandardParallel1())
	{
		itsLogger.Trace("Standard latitude 1 does not match: " + to_string(StandardParallel1()) + " vs " +
		                to_string(other.StandardParallel1()));
		return false;
	}

	if (StandardParallel2() != other.StandardParallel2())
	{
		itsLogger.Trace("Standard latitude 2 does not match: " + to_string(StandardParallel2()) + " vs " +
		                to_string(other.StandardParallel2()));
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
	regular_grid::Write(file);

	file << "__itsOrientation__ " << Orientation() << endl
	     << "__itsStandardParallel1__ " << StandardParallel1() << endl
	     << "__itsStandardParallel2__ " << StandardParallel2() << endl;

	return file;
}

double lambert_conformal_grid::StandardParallel1() const
{
	return itsSpatialReference->GetProjParm(SRS_PP_STANDARD_PARALLEL_1, 0.0);
}
double lambert_conformal_grid::StandardParallel2() const
{
	return itsSpatialReference->GetProjParm(SRS_PP_STANDARD_PARALLEL_2, 0.0);
}

OGRSpatialReference lambert_conformal_grid::SpatialReference() const
{
	return OGRSpatialReference(*itsSpatialReference);
}

double lambert_conformal_grid::Cone() const
{
	const double sp1 = StandardParallel1(), sp2 = StandardParallel2();

	if (fabs(sp1 - sp2) < 0.0001)
	{
		return sin(fabs(sp1) * constants::kDeg);
	}

	return (log(cos(sp1 * constants::kDeg)) - log(cos(sp2 * constants::kDeg))) /
	       (log(tan((90 - fabs(sp1)) * constants::kDeg * 0.5)) - log(tan(90 - fabs(sp2)) * constants::kDeg * 0.5));
}
