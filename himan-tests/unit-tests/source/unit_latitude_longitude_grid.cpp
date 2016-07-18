#define BOOST_TEST_DYN_LINK

#include "himan_unit.h"
#include "latitude_longitude_grid.h"

#define BOOST_TEST_MODULE latitude_longitude_grid

using namespace std;
using namespace himan;

const double kEpsilon = 1e-3;

BOOST_AUTO_TEST_CASE(NI)
{
	latitude_longitude_grid lg;
	lg.Ni(100);

	BOOST_REQUIRE(lg.Ni() == 100);

}

BOOST_AUTO_TEST_CASE(NJ)
{
	latitude_longitude_grid lg;
	lg.Nj(100);

	BOOST_REQUIRE(lg.Nj() == 100);

}

BOOST_AUTO_TEST_CASE(SCANNING_MODE)
{
	latitude_longitude_grid lg;
	lg.ScanningMode(kTopLeft);

	BOOST_REQUIRE(lg.ScanningMode() == kTopLeft);
}

BOOST_AUTO_TEST_CASE(COORDINATES)
{
	latitude_longitude_grid lg;
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
	latitude_longitude_grid lg;

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
	latitude_longitude_grid lg1;
	lg1.Ni(100);

	latitude_longitude_grid lg2;
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


}

BOOST_AUTO_TEST_CASE(LATLON)
{
	latitude_longitude_grid lg;

	lg.ScanningMode(kTopLeft);
	lg.BottomLeft(point(25,68));
	lg.TopRight(point(33,75));

	lg.Ni(8);
	lg.Nj(8);

	BOOST_REQUIRE(lg.LatLon(0) == point(25,75));
	auto pt = lg.LatLon(1);
	BOOST_CHECK_CLOSE(pt.X(), 26.1428, kEpsilon);
	BOOST_REQUIRE(pt.Y() == 75);

	pt = lg.LatLon(18);
	BOOST_CHECK_CLOSE(pt.X(), 27.2857, kEpsilon);
	BOOST_REQUIRE(pt.Y() == 73);

	lg.ScanningMode(kBottomLeft);
	BOOST_REQUIRE(lg.LatLon(0) == point(25,68));
}

BOOST_AUTO_TEST_CASE(SWAP)
{

	latitude_longitude_grid g(kBottomLeft, point(10,40), point(15,45));

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
