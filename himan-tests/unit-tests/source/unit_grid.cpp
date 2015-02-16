//#define BOOST_TEST_DYN_LINK

#include "himan_unit.h"
#include "logger.h"
#include "regular_grid.h"
#include "point.h"

#define BOOST_TEST_MODULE grid

using namespace std;
using namespace himan;

BOOST_AUTO_TEST_CASE(AREA_CORNERS)
{
	regular_grid g(kBottomLeft,
			true,
			kLatLonProjection,
			point(30,60),
			point(40,80),
			point(),
			0);

	BOOST_REQUIRE(g.BottomLeft() == point(30,60));
	BOOST_REQUIRE(g.TopRight() == point(40,80));
	BOOST_REQUIRE(g.BottomRight() == point(40,60));
	BOOST_REQUIRE(g.TopLeft() == point(30,80));
}

BOOST_AUTO_TEST_CASE(LATLON)
{

	regular_grid g(kBottomLeft,
			true,
			kLatLonProjection,
			point(10,40),
			point(20,50),
			point(),
			0);

	g.Di(0.5);
	g.Dj(0.5);

	g.Ni(21);
	g.Nj(21);

	BOOST_REQUIRE(g.FirstGridPoint() == point(10,40));

	BOOST_REQUIRE(g.LatLon(0) == point(10,40));
	BOOST_REQUIRE(g.LatLon(1) == point(10.5,40));
	BOOST_REQUIRE(g.LatLon(22) == point(10.5,40.5));
	BOOST_REQUIRE(g.LatLon(440) == point(20,50));

	regular_grid g2 = regular_grid(kTopLeft,
			true,
			kLatLonProjection,
			point(-40,-10),
			point(0,10),
			point(),
			0);

	g2.Di(5);
	g2.Dj(2);

	g2.Ni(9);
	g2.Nj(11);

	BOOST_REQUIRE(g2.FirstGridPoint() == point(-40, 10));

	BOOST_REQUIRE(g2.LatLon(0) == point(-40, 10));
	BOOST_REQUIRE(g2.LatLon(1) == point(-35, 10));
	BOOST_REQUIRE(g2.LatLon(9) == point(-40,8));
	BOOST_REQUIRE(g2.LatLon(98) == point(0, -10));

}

BOOST_AUTO_TEST_CASE(SWAP)
{

	regular_grid g(kBottomLeft,
			true,
			kRotatedLatLonProjection,
			point(10,40),
			point(15,45),
			point(10, -30),
			0);

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

BOOST_AUTO_TEST_CASE(FIRST_LAST_GRIDPOINT)
{

	regular_grid g(kBottomLeft,
			true,
			kLatLonProjection,
			point(10,40),
			point(15,45),
			point(),
			0);

	g.Di(1);
	g.Dj(1);

	g.Ni(6);
	g.Nj(6);

	BOOST_REQUIRE(g.FirstGridPoint() == point(10,40));
	BOOST_REQUIRE(g.LastGridPoint() == point(15,45));

	g.Swap(kTopLeft);

	BOOST_REQUIRE(g.FirstGridPoint() == g.TopLeft());
	BOOST_REQUIRE(g.LastGridPoint() == g.BottomRight());

}