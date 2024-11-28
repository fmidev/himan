/**
 * @file grid.cpp
 *
 */

#include "grid.h"
#include "util.h"
#include <cpl_conv.h>
#include <ogr_geometry.h>
#include <ogr_spatialref.h>

using namespace himan;
using namespace std;

grid::grid(HPGridClass gridClass, HPGridType gridType, bool uvRelativeToGrid, const std::string& theName)
    : itsGridClass(gridClass), itsGridType(gridType), itsUVRelativeToGrid(uvRelativeToGrid), itsName(theName)
{
}

bool grid::EqualsTo(const grid& other) const
{
	if (itsName.empty() == false && itsName == other.itsName)
	{
		return true;
	}

	if (other.itsGridType != itsGridType)
	{
		itsLogger.Trace(fmt::format("Grid type does not match: {} vs {}", HPGridTypeToString.at(itsGridType),
		                            HPGridTypeToString.at(other.Type())));
		return false;
	}

	if (other.itsGridClass != itsGridClass)
	{
		itsLogger.Trace(fmt::format("Grid class does not match: {} vs {}", HPGridClassToString.at(itsGridClass),
		                            HPGridClassToString.at(other.itsGridClass)));
		return false;
	}

	return true;
}

HPGridType grid::Type() const
{
	return itsGridType;
}

HPGridClass grid::Class() const
{
	return itsGridClass;
}

std::string grid::Name() const
{
	return itsName;
}

ostream& grid::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;

	file << "__itsUVRelativeToGrid__ " << itsUVRelativeToGrid << std::endl;
	file << "__itsName__ " << itsName << std::endl;
	file << EarthShape();

	return file;
}

size_t grid::Size() const
{
	throw runtime_error("grid::Size() called");
}
point grid::LatLon(size_t theLocationIndex) const
{
	throw runtime_error("grid::LatLon() called");
}
bool grid::operator!=(const grid& other) const
{
	return !(other == *this);
}
bool grid::operator==(const grid& other) const
{
	return EqualsTo(other);
}
bool grid::UVRelativeToGrid() const
{
	return itsUVRelativeToGrid;
}
void grid::UVRelativeToGrid(bool theUVRelativeToGrid)
{
	itsUVRelativeToGrid = theUVRelativeToGrid;
}
std::vector<point> grid::GridPointsInProjectionSpace() const
{
	throw runtime_error("grid::GridPointsInProjectionSpace() called");
}

//--------------- regular grid

regular_grid::regular_grid(HPGridType gridType, HPScanningMode scMode, double di, double dj, size_t ni, size_t nj,
                           bool uvRelativeToGrid, const std::string& theName)
    : grid(kRegularGrid, gridType, uvRelativeToGrid, theName),
      itsScanningMode(scMode),
      itsDi(di),
      itsDj(dj),
      itsNi(ni),
      itsNj(nj)
{
}

regular_grid::regular_grid(const regular_grid& other)
    : grid(other),
      itsScanningMode(other.itsScanningMode),
      itsDi(other.itsDi),
      itsDj(other.itsDj),
      itsNi(other.itsNi),
      itsNj(other.itsNj)
{
}

bool regular_grid::EqualsTo(const regular_grid& other) const
{
	if (!grid::EqualsTo(other))
	{
		return false;
	}

	// Turned off for now
	if (false && other.itsScanningMode != itsScanningMode)
	{
		return false;
	}

#if 0
	const auto es = EarthShape();
	const auto oes = other.EarthShape();

	// Turned off for now
	if (false && es != oes)
	{
		itsLogger.Trace("Earth shape does not match: " + to_string(es.A()) + "/" + to_string(es.B()) + " vs " +
		                to_string(oes.A()) + "/" + to_string(oes.B()));

		return false;
	}
#endif

	const double kEpsilon = 0.0001;

	if (fabs(other.itsDi - itsDi) > kEpsilon)
	{
		itsLogger.Trace(fmt::format("Di does not match: {} vs {} (diff: {})", itsDi, other.itsDi, itsDi - other.itsDi));
		return false;
	}

	if (fabs(other.itsDj - itsDj) > kEpsilon)
	{
		itsLogger.Trace(fmt::format("Dj does not match: {} vs {} (diff: {})", itsDj, other.itsDj, itsDj - other.itsDj));
		return false;
	}

	if (other.itsNi != itsNi)
	{
		itsLogger.Trace(fmt::format("Ni does not match: {} vs {}", itsNi, other.itsNi));
		return false;
	}

	if (other.itsNj != itsNj)
	{
		itsLogger.Trace(fmt::format("Nj does not match: {} vs {}", itsNj, other.itsNj));
		return false;
	}

	// Regular grids are always defined in latlon so use that comparison function here

	if (!point::LatLonCompare(other.BottomLeft(), BottomLeft()))
	{
		itsLogger.Trace(fmt::format("BottomLeft does not match: {} vs {}", static_cast<std::string>(other.BottomLeft()),
		                            static_cast<std::string>(BottomLeft())));
		return false;
	}

	if (!point::LatLonCompare(other.TopLeft(), TopLeft()))
	{
		itsLogger.Trace(fmt::format("TopLeft does not match: {} vs {}", static_cast<std::string>(other.TopLeft()),
		                            static_cast<std::string>(TopLeft())));
		return false;
	}

	if (!point::LatLonCompare(other.BottomRight(), BottomRight()))
	{
		itsLogger.Trace(fmt::format("BottomRight does not match: {} vs {}",
		                            static_cast<std::string>(other.BottomRight()),
		                            static_cast<std::string>(BottomRight())));
		return false;
	}

	if (!point::LatLonCompare(other.TopRight(), TopRight()))
	{
		itsLogger.Trace(fmt::format("TopRight does not match: {} vs {}", static_cast<std::string>(other.TopRight()),
		                            static_cast<std::string>(TopRight())));
		return false;
	}

	return true;
}

HPScanningMode regular_grid::ScanningMode() const
{
	return itsScanningMode;
}

std::string regular_grid::Proj4String() const
{
	if (itsSpatialReference == nullptr)
	{
		throw std::runtime_error("Spatial reference instance not initialized");
	}

	char* projstr;
	if (itsSpatialReference->exportToProj4(&projstr) != OGRERR_NONE)
	{
		throw std::runtime_error("Failed to get proj4 str");
	}

	std::string proj(projstr);
	CPLFree(projstr);
	return proj;
}

std::string regular_grid::WKT(const std::map<std::string, std::string>& opts) const
{
	if (itsSpatialReference == nullptr)
	{
		throw std::runtime_error("Spatial reference instance not initialized");
	}

	char** projopts = nullptr;

	// https://gdal.org/doxygen/classOGRSpatialReference.html#ae986da88649783b5c194de55c46890a5
	for (const auto& opt : opts)
	{
		projopts = CSLAddNameValue(projopts, opt.first.c_str(), opt.second.c_str());
	}

	char* wkt;
	if (itsSpatialReference->exportToWkt(&wkt) != OGRERR_NONE)
	{
		throw std::runtime_error("Failed to get wkt");
	}

	std::string w(wkt);
	CPLFree(wkt);
	CSLDestroy(projopts);

	return w;
}

point regular_grid::Projected(const point& latlon) const
{
	double projX = latlon.X(), projY = latlon.Y();
	ASSERT(itsLatLonToXYTransformer);

	// Transform latlon to projected coordinates.
	// Projected coordinates are in meters, with false easting and
	// false northing applied so that point 0,0 is top left or bottom left,
	// depending on the scanning mode.

	if (!itsLatLonToXYTransformer->Transform(1, &projX, &projY))
	{
		itsLogger.Error(fmt::format("Error determining xy value for latlon point {},{}", latlon.X(), latlon.Y()));
		return point();
	}

	return point(projX, projY);
}

point regular_grid::XY(const point& latlon) const
{
	// 1. Get latlon point in projected space
	const point proj = Projected(latlon);

	// 2. Transform projected coordinates (meters) to grid xy (no unit).
	// Projected coordinates run from 0 ... area width and 0 ... area height.
	// Grid point coordinates run from 0 ... ni and 0 ... nj.

	const double x = (proj.X() / itsDi);
	const double y = (proj.Y() / itsDj) * (itsScanningMode == kTopLeft ? -1 : 1);

	if (x < 0. || x > static_cast<double>(itsNi - 1) || y < 0. || y > static_cast<double>(itsNj - 1))
	{
		return point(MissingDouble(), MissingDouble());
	}

	return point(x, y);
}

point regular_grid::LatLon(const point& projected) const
{
	double x = projected.X();
	double y = projected.Y();

	ASSERT(itsXYToLatLonTransformer);
	if (!itsXYToLatLonTransformer->Transform(1, &x, &y))
	{
		itsLogger.Error(fmt::format("Error determining latitude longitude value for xy point {},{}", x, y));
		return point();
	}

	return point(x, y);
}

point regular_grid::LatLon(size_t locationIndex) const
{
	ASSERT(IsValid(itsDi));
	ASSERT(IsValid(itsDj));
	ASSERT(locationIndex < itsNi * itsNj);

	const size_t jIndex = static_cast<size_t>(locationIndex / itsNi);
	const size_t iIndex = static_cast<size_t>(locationIndex % itsNi);

	double x = static_cast<double>(iIndex) * itsDi;
	double y = static_cast<double>(jIndex) * itsDj * (itsScanningMode == kTopLeft ? -1 : 1);

	return LatLon(point(x, y));
}

size_t regular_grid::Size() const
{
	return itsNi * itsNj;
}

point regular_grid::FirstPoint() const
{
	return LatLon(0);
}

point regular_grid::LastPoint() const
{
	return LatLon(Size() - 1);
}

point regular_grid::BottomLeft() const
{
	switch (itsScanningMode)
	{
		case kBottomLeft:
			return LatLon(0);
		case kTopLeft:
			return LatLon(itsNj * itsNi - itsNi);
		default:
			itsLogger.Fatal(fmt::format("Unhandled scanning mode: {}", HPScanningModeToString.at(itsScanningMode)));
			himan::Abort();
	}
}
point regular_grid::TopRight() const
{
	switch (itsScanningMode)
	{
		case kBottomLeft:
			return LatLon(itsNj * itsNi - 1);
		case kTopLeft:
			return LatLon(itsNi - 1);
		default:
			itsLogger.Fatal(fmt::format("Unhandled scanning mode: {}", HPScanningModeToString.at(itsScanningMode)));
			himan::Abort();
	}
}
point regular_grid::TopLeft() const
{
	switch (itsScanningMode)
	{
		case kBottomLeft:
			return LatLon(itsNj * itsNi - itsNi);
		case kTopLeft:
			return LatLon(0);
		default:
			itsLogger.Fatal(fmt::format("Unhandled scanning mode: {}", HPScanningModeToString.at(itsScanningMode)));
			himan::Abort();
	}
}
point regular_grid::BottomRight() const
{
	switch (itsScanningMode)
	{
		case kBottomLeft:
			return LatLon(itsNi - 1);
		case kTopLeft:
			return LatLon(itsNi * itsNj - 1);
		default:
			itsLogger.Fatal(fmt::format("Unhandled scanning mode: {}", HPScanningModeToString.at(itsScanningMode)));
			himan::Abort();
	}
}

size_t regular_grid::Ni() const
{
	return itsNi;
}
size_t regular_grid::Nj() const
{
	return itsNj;
}
double regular_grid::Di() const
{
	return itsDi;
}
double regular_grid::Dj() const
{
	return itsDj;
}
earth_shape<double> regular_grid::EarthShape() const
{
	// reverse mapping from OGRSpatialReference to himan::earth_shape

	OGRErr err;
	const double A = itsSpatialReference->GetSemiMajor(&err);

	if (err != OGRERR_NONE)
	{
		itsLogger.Fatal("Unable to get Semi Major");
		himan::Abort();
	}

	const double B = itsSpatialReference->GetSemiMinor(&err);

	if (err != OGRERR_NONE)
	{
		itsLogger.Fatal("Unable to get Semi Minor");
		himan::Abort();
	}

	int epsg = itsSpatialReference->GetEPSGGeogCS();

	std::string name;
	if (A == B && A == 6371220)
	{
		name = "newbase";
	}
	else if (epsg == 4326)
	{
		name = "WGS84";
	}

	return earth_shape<double>(A, B, name);
}

ostream& regular_grid::Write(std::ostream& file) const
{
	grid::Write(file);

	file << "<" << ClassName() << ">" << std::endl
	     << "__itsScanningMode__ " << HPScanningModeToString.at(itsScanningMode) << std::endl
	     << "__itsDi__ " << itsDi << std::endl
	     << "__itsDj__ " << itsDj << std::endl
	     << "__itsNi__ " << itsNi << std::endl
	     << "__itsNj__ " << itsNj << std::endl;

	return file;
}

std::unique_ptr<OGRPolygon> regular_grid::Geometry() const
{
	OGRLinearRing ring;

	auto Index = [&](size_t x, size_t y) -> size_t { return y * itsNi + x; };

	// A         B
	//  +-------+
	//  |       |
	//  |       |
	//  +-------+
	// C         D

	// assuming topleft (works the same of bottomleft)
	// AB

	for (size_t i = 0; i < itsNi; i++)
	{
		const point p = LatLon(Index(i, 0));
		ring.addPoint(p.X(), p.Y());
	}

	// BD
	for (size_t j = 0; j < itsNj; j++)
	{
		const point p = LatLon(Index(itsNi - 1, j));
		ring.addPoint(p.X(), p.Y());
	}

	// CD
	for (int i = static_cast<int>(itsNi) - 1; i >= 0; --i)
	{
		const point p = LatLon(Index(i, itsNj - 1));
		ring.addPoint(p.X(), p.Y());
	}

	// AC
	for (int j = static_cast<int>(itsNj) - 1; j >= 0; --j)
	{
		const point p = LatLon(Index(0, j));
		ring.addPoint(p.X(), p.Y());
	}

	auto geometry = std::unique_ptr<OGRPolygon>(new OGRPolygon());
	geometry->addRing(&ring);
	return geometry;
}

std::unique_ptr<OGRSpatialReference> regular_grid::SpatialReference() const
{
	return std::unique_ptr<OGRSpatialReference>(itsSpatialReference->Clone());
}

std::vector<point> regular_grid::GridPointsInProjectionSpace() const
{
	std::vector<point> ret;
	ret.reserve(Size());

	point first = Projected(FirstPoint());

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

std::vector<point> regular_grid::XY(const regular_grid& target) const
{
	// 1. Create list of points in the projection space of the
	// target grid.

	const auto targetProj = target.GridPointsInProjectionSpace();

	// 2. Transform the points to the projection space of the source
	// grid

	auto tosp = target.SpatialReference();
	const point soff = Projected(FirstPoint());

	vector<point> sourceProj;

	if (itsSpatialReference->IsSame(tosp.get()))
	{
		itsLogger.Trace("Spatial references are equal, no need to do transformation");
		sourceProj = targetProj;
	}
	else
	{
		sourceProj.reserve(targetProj.size());

		auto xform = std::unique_ptr<OGRCoordinateTransformation>(
		    OGRCreateCoordinateTransformation(target.SpatialReference().get(), itsSpatialReference.get()));

		for (const auto& p : targetProj)
		{
			double x = p.X(), y = p.Y();

			if (!xform->Transform(1, &x, &y))
				himan::Abort();
			sourceProj.emplace_back(x, y);
		}
	}

	// 3. Transform projected coordinates to grid space

	const double ni = static_cast<double>(itsNi - 1);
	const double nj = static_cast<double>(itsNj - 1);
	const double di = itsDi;
	const double dj = itsDj * (itsScanningMode == kTopLeft ? -1 : 1);

	std::vector<point> sourceXY;
	sourceXY.reserve(sourceProj.size());

	for (auto& p : sourceProj)
	{
		double x = (p.X() - soff.X()) / di, y = (p.Y() - soff.Y()) / dj;

		if (x < 0. || x > ni || y < 0. || y > nj)
		{
			x = y = MissingDouble();
		}
		sourceXY.emplace_back(x, y);
	}
	return sourceXY;
}

std::pair<double, double> regular_grid::FalseEastingAndNorthing(const OGRSpatialReference* spRef,
                                                                OGRCoordinateTransformation* llToProjXForm,
                                                                const point& firstPoint, bool firstPointIsProjected)
{
	// First get first point coordinates in projected space. Transform from
	// latlon if necessary.
	double lat = firstPoint.Y(), lon = firstPoint.X();

	if (firstPointIsProjected == false)
	{
		if (!llToProjXForm->Transform(1, &lon, &lat))
		{
			logger logr("regular_grid");
			logr.Error("Failed to get false easting and northing");
			return std::make_pair(MissingDouble(), MissingDouble());
		}
	}

	// HIMAN-336: limit false easting/northing accuracy to four decimal places (millimeters)

	lon = util::round(lon, 4);
	lat = util::round(lat, 4);

	if (fabs(lon) < 1e-4 and fabs(lat) < 1e-4)
	{
		return std::make_pair(0.0, 0.0);
	}

	double fe = spRef->GetProjParm(SRS_PP_FALSE_EASTING, 0.0) - lon;
	double fn = spRef->GetProjParm(SRS_PP_FALSE_NORTHING, 0.0) - lat;

	return std::make_pair(fe, fn);
}

//--------------- irregular grid

irregular_grid::irregular_grid(HPGridType type) : grid(kIrregularGrid, type, false)
{
}

irregular_grid::irregular_grid(const irregular_grid& other) : grid(other)
{
}

bool irregular_grid::EqualsTo(const irregular_grid& other) const
{
	if (!grid::EqualsTo(other))
	{
		return false;
	}

	return true;
}

earth_shape<double> irregular_grid::EarthShape() const
{
	return itsEarthShape;
}

void irregular_grid::EarthShape(const earth_shape<double>& theShape)
{
	itsEarthShape = theShape;
}
