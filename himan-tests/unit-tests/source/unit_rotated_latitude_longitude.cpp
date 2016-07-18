#define BOOST_TEST_DYN_LINK

#include "himan_unit.h"
#include "latitude_longitude_grid.h"

#define BOOST_TEST_MODULE rotated_latitude_longitude_grid

using namespace std;
using namespace himan;

const double kEpsilon = 1e-3;

BOOST_AUTO_TEST_CASE(NI)
{
	rotated_latitude_longitude_grid lg;
	lg.Ni(100);

	BOOST_REQUIRE(lg.Ni() == 100);

}

BOOST_AUTO_TEST_CASE(NJ)
{
	rotated_latitude_longitude_grid lg;
	lg.Nj(100);

	BOOST_REQUIRE(lg.Nj() == 100);

}

BOOST_AUTO_TEST_CASE(SOUTHPOLE)
{
	rotated_latitude_longitude_grid lg;
	lg.SouthPole(point(0, -40));

	BOOST_REQUIRE(lg.SouthPole() == point(0,-40));

}

BOOST_AUTO_TEST_CASE(UVRELATIVETOGRID)
{
	rotated_latitude_longitude_grid lg;

	BOOST_REQUIRE(!lg.UVRelativeToGrid());

	lg.UVRelativeToGrid(true);

	BOOST_REQUIRE(lg.UVRelativeToGrid());
}

BOOST_AUTO_TEST_CASE(SCANNING_MODE)
{
	rotated_latitude_longitude_grid lg;
	lg.ScanningMode(kTopLeft);

	BOOST_REQUIRE(lg.ScanningMode() == kTopLeft);
}

BOOST_AUTO_TEST_CASE(COORDINATES)
{
	rotated_latitude_longitude_grid lg;
	lg.BottomLeft(point(-180,-90));

	BOOST_REQUIRE(lg.BottomLeft() == point(-180,-90));

	lg.TopRight(point(180,90));
	BOOST_REQUIRE(lg.BottomLeft() == point(-180,-90));
	BOOST_REQUIRE(lg.TopRight() == point(180,90));
	BOOST_REQUIRE(lg.BottomRight() == point(180,-90));
	BOOST_REQUIRE(lg.TopLeft() == point(-180,90));

}

BOOST_AUTO_TEST_CASE(FIRST_LAST_POINT)
{
	rotated_latitude_longitude_grid lg;

	lg.TopRight(point(180,90));
	lg.BottomLeft(point(-180,-90));
	lg.ScanningMode(kBottomLeft);

	BOOST_REQUIRE(lg.FirstPoint() == point(-180,-90));
	BOOST_REQUIRE(lg.LastPoint() == point(180,90));

	lg.ScanningMode(kTopLeft);

	BOOST_REQUIRE(lg.FirstPoint() == point(-180,90));
	BOOST_REQUIRE(lg.LastPoint() == point(180,-90));

}

BOOST_AUTO_TEST_CASE(EQUALITY)
{
	rotated_latitude_longitude_grid lg1;
	lg1.Ni(100);

	rotated_latitude_longitude_grid lg2;
	lg2.Ni(101);
	
	BOOST_REQUIRE(lg1 != lg2);

	lg2.Ni(100);
	lg1.Nj(200);
	lg2.Nj(200);

	BOOST_REQUIRE(lg1 == lg2);

	lg1.BottomLeft(point(25,60));
	lg2.BottomLeft(point(25,60));
	lg1.TopRight(point(30,70));
	lg2.TopRight(point(30,71));

	BOOST_REQUIRE(lg1 != lg2);

	lg2.TopRight(point(30,70));

	BOOST_REQUIRE(lg1 == lg2);

	lg2.UVRelativeToGrid(true);

	BOOST_REQUIRE(lg1 != lg2);

}

BOOST_AUTO_TEST_CASE(LATLON)
{
	rotated_latitude_longitude_grid lg;

	lg.ScanningMode(kTopLeft);
	lg.BottomLeft(point(25,68));
	lg.TopRight(point(33,75));

	lg.Ni(8);
	lg.Nj(8);

	BOOST_REQUIRE(lg.LatLon(0) == point(106, 75));
	auto pt = lg.LatLon(1);

	BOOST_CHECK_CLOSE(pt.X(), 107.143, kEpsilon);
	BOOST_CHECK_CLOSE(pt.Y(), 75, kEpsilon);

	pt = lg.LatLon(18);
	BOOST_CHECK_CLOSE(pt.X(), 108.286, kEpsilon);
	BOOST_CHECK_CLOSE(pt.Y(), 73, kEpsilon);

	lg.ScanningMode(kBottomLeft);
	BOOST_REQUIRE(lg.LatLon(0) == point(106, 68));
}

BOOST_AUTO_TEST_CASE(SWAP)
{

	rotated_latitude_longitude_grid g(kBottomLeft, point(10,40), point(15,45), point(0,-30));

	g.Data().Resize(6, 6, 1);

	g.Di(1);
	g.Dj(1);

	g.Ni(6);
	g.Nj(6);

	double j = 0;

	for (size_t i = 1; i <= g.Size(); i++)
	{
		g.Value(i-1, j);

		if (i % g.Nj() == 0) j++;
	}
#if 0
	for (size_t i = 1; i <= g.Size(); i++)
	{
		cout << g.Value(i-1) << " ";

		if (i % g.Nj() == 0) cout << "\n";
	}
#endif

	BOOST_REQUIRE(g.Value(0) == 0);
	BOOST_REQUIRE(g.Value(35) == 5);
	
	g.Swap(kTopLeft);

	BOOST_REQUIRE(g.Value(0) == 5);
	BOOST_REQUIRE(g.Value(35) == 0);

	g.Swap(kBottomLeft);

	BOOST_REQUIRE(g.Value(0) == 0);
	BOOST_REQUIRE(g.Value(35) == 5);
		
}
