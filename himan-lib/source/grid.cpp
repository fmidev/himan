/**
 * @file grid.cpp
 *
 */

#include "grid.h"
#include <cpl_conv.h>
#include <ogr_geometry.h>
#include <ogr_spatialref.h>

using namespace himan;
using namespace std;

grid::grid(HPGridClass gridClass, HPGridType gridType, bool uvRelativeToGrid)
    : itsGridClass(gridClass), itsGridType(gridType), itsUVRelativeToGrid(uvRelativeToGrid)
{
}

bool grid::EqualsTo(const grid& other) const
{
	if (other.itsGridType != itsGridType)
	{
		itsLogger.Trace("Grid type does not match: " + HPGridTypeToString.at(itsGridType) + " vs " +
		                HPGridTypeToString.at(other.Type()));
		return false;
	}

	if (other.itsGridClass != itsGridClass)
	{
		itsLogger.Trace("Grid class does not match: " + HPGridClassToString.at(itsGridClass) + " vs " +
		                HPGridClassToString.at(other.itsGridClass));
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

ostream& grid::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;

	file << "__itsUVRelativeToGrid__ " << itsUVRelativeToGrid << std::endl;

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
                           bool uvRelativeToGrid)
    : grid(kRegularGrid, gridType, uvRelativeToGrid),
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
		itsLogger.Trace("Di does not match: " + to_string(itsDi) + " vs " + to_string(other.itsDi));
		return false;
	}

	if ((other.itsDj != itsDj) > kEpsilon)
	{
		itsLogger.Trace("Dj does not match: " + to_string(itsDj) + " vs " + to_string(other.itsDj));
		return false;
	}

	if (other.itsNi != itsNi)
	{
		itsLogger.Trace("Ni does not match: " + to_string(itsNi) + " vs " + to_string(other.itsNi));
		return false;
	}

	if (other.itsNj != itsNj)
	{
		itsLogger.Trace("Nj does not match: " + to_string(itsNj) + " vs " + to_string(other.itsNj));
		return false;
	}

	// Regular grids are always defined in latlon so use that comparison function here

	if (!point::LatLonCompare(other.BottomLeft(), BottomLeft()))
	{
		itsLogger.Trace("BottomLeft does not match: " + static_cast<std::string>(other.BottomLeft()) + " vs " +
		                static_cast<std::string>(BottomLeft()));
		return false;
	}

	if (!point::LatLonCompare(other.TopLeft(), TopLeft()))
	{
		itsLogger.Trace("TopLeft does not match: " + static_cast<std::string>(other.TopLeft()) + " vs " +
		                static_cast<std::string>(TopLeft()));
		return false;
	}

	if (!point::LatLonCompare(other.BottomRight(), BottomRight()))
	{
		itsLogger.Trace("BottomRight does not match: " + static_cast<std::string>(other.BottomRight()) + " vs " +
		                static_cast<std::string>(BottomRight()));
		return false;
	}

	if (!point::LatLonCompare(other.TopRight(), TopRight()))
	{
		itsLogger.Trace("TopRight does not match: " + static_cast<std::string>(other.TopRight()) + " vs " +
		                static_cast<std::string>(TopRight()));
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
		itsLogger.Error("Error determining xy value for latlon point " + std::to_string(latlon.X()) + "," +
		                std::to_string(latlon.Y()));
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
		itsLogger.Error("Error determining latitude longitude value for xy point " + std::to_string(x) + "," +
		                std::to_string(y));
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

	return earth_shape<double>(A, B);
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
	return std::move(geometry);
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
