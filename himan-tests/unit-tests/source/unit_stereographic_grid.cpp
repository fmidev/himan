#define BOOST_TEST_DYN_LINK

#include "himan_unit.h"
#include "stereographic_grid.h"

#define BOOST_TEST_MODULE stereographic_grid

using namespace std;
using namespace himan;

const double kEpsilon = 1e-3;

BOOST_AUTO_TEST_CASE(NI)
{
	stereographic_grid sg;
	sg.Ni(100);

	BOOST_REQUIRE(sg.Ni() == 100);

}

BOOST_AUTO_TEST_CASE(NJ)
{
	stereographic_grid sg;
	sg.Nj(100);

	BOOST_REQUIRE(sg.Nj() == 100);

}

BOOST_AUTO_TEST_CASE(ORIENTATION)
{
	stereographic_grid sg;
	sg.Orientation(60);

	BOOST_REQUIRE(sg.Orientation() == 60);

}

BOOST_AUTO_TEST_CASE(SCANNING_MODE)
{
	stereographic_grid sg;
	sg.ScanningMode(kBottomLeft);

	BOOST_REQUIRE(sg.ScanningMode() == kBottomLeft);
}

BOOST_AUTO_TEST_CASE(COORDINATES)
{
	stereographic_grid sg;
	sg.BottomLeft(point(-180,-90));

	BOOST_REQUIRE(sg.BottomLeft() == point(-180,-90));

	sg.TopRight(point(180,90));
	BOOST_REQUIRE(sg.BottomLeft() == point(-180,-90));
	BOOST_REQUIRE(sg.TopRight() == point(180,90));
	

}

BOOST_AUTO_TEST_CASE(FIRST_LAST_POINT)
{
	stereographic_grid sg;

	sg.TopRight(point(180,90));
	sg.BottomLeft(point(-180,-90));
	sg.ScanningMode(kBottomLeft);

	BOOST_REQUIRE(sg.FirstPoint() == point(-180,-90));
	BOOST_REQUIRE(sg.LastPoint() == point(180,90));

}

BOOST_AUTO_TEST_CASE(EQUALITY)
{
	stereographic_grid sg1;
	sg1.Ni(100);

	stereographic_grid sg2;
	sg2.Ni(101);
	
	BOOST_REQUIRE(sg1 != sg2);

	sg2.Ni(100);
	sg1.Nj(200);
	sg2.Nj(200);

	BOOST_REQUIRE(sg1 == sg2);

	sg1.BottomLeft(point(25,60));
	sg2.BottomLeft(point(25,60));
	sg1.TopRight(point(30,70));
	sg2.TopRight(point(30,71));

	BOOST_REQUIRE(sg1 != sg2);

	sg2.TopRight(point(30,70));

	BOOST_REQUIRE(sg1 == sg2);

	sg2.Orientation(10);

	BOOST_REQUIRE(sg1 != sg2);

}

/*
BOOST_AUTO_TEST_CASE(LATLON)
{
	stereographic_grid sg;

	sg.ScanningMode(kBottomLeft);
	sg.BottomLeft(point(25,68));
	sg.TopRight(point(33,75));

	sg.Ni(8);
	sg.Nj(8);

	BOOST_REQUIRE(sg.LatLon(0) == point(25, 68));
	auto pt = sg.LatLon(1);

	BOOST_CHECK_CLOSE(pt.X(), 107.143, kEpsilon);
	BOOST_CHECK_CLOSE(pt.Y(), 75, kEpsilon);

	pt = sg.LatLon(18);
	BOOST_CHECK_CLOSE(pt.X(), 108.286, kEpsilon);
	BOOST_CHECK_CLOSE(pt.Y(), 73, kEpsilon);

	sg.ScanningMode(kBottomLeft);
	BOOST_REQUIRE(sg.LatLon(0) == point(106, 68));
}
*/

BOOST_AUTO_TEST_CASE(SWAP)
{

	stereographic_grid g(kBottomLeft, point(10,40), point(15,45), 60);

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

	BOOST_REQUIRE(g.Value(0) == 0);
	BOOST_REQUIRE(g.Value(35) == 5);
	
	g.Swap(kTopLeft);

	BOOST_REQUIRE(g.Value(0) == 5);
	BOOST_REQUIRE(g.Value(35) == 0);

	g.Swap(kBottomLeft);

	BOOST_REQUIRE(g.Value(0) == 0);
	BOOST_REQUIRE(g.Value(35) == 5);
		
}
