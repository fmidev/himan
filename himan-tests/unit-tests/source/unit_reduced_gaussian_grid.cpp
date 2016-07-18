#define BOOST_TEST_DYN_LINK

#include "himan_unit.h"
#include "reduced_gaussian_grid.h"

#define BOOST_TEST_MODULE reduced_gaussian_grid

using namespace std;
using namespace himan;

const double kEpsilon = 1e-3;

BOOST_AUTO_TEST_CASE(N)
{
	reduced_gaussian_grid rg;
	rg.N(100);

	BOOST_REQUIRE(rg.N() == 100);

}

BOOST_AUTO_TEST_CASE(NJ)
{
	reduced_gaussian_grid rg;
	rg.Nj(100);

	BOOST_REQUIRE(rg.Nj() == 100);

}

BOOST_AUTO_TEST_CASE(SCANNING_MODE)
{
	reduced_gaussian_grid rg;
	rg.ScanningMode(kTopLeft);

	BOOST_REQUIRE(rg.ScanningMode() == kTopLeft);
}

BOOST_AUTO_TEST_CASE(COORDINATES)
{
	reduced_gaussian_grid rg;
	rg.BottomLeft(point(-180,-90));

	BOOST_REQUIRE(rg.BottomLeft() == point(-180,-90));

	rg.TopRight(point(180,90));
	BOOST_REQUIRE(rg.BottomLeft() == point(-180,-90));
	BOOST_REQUIRE(rg.TopRight() == point(180,90));
	BOOST_REQUIRE(rg.BottomRight() == point(180,-90));
	BOOST_REQUIRE(rg.TopLeft() == point(-180,90));

}

BOOST_AUTO_TEST_CASE(FIRST_LAST_POINT)
{
	reduced_gaussian_grid rg;

	rg.TopRight(point(180,90));
	rg.BottomLeft(point(-180,-90));
	rg.ScanningMode(kBottomLeft);

	BOOST_REQUIRE(rg.FirstPoint() == point(-180,-90));
	BOOST_REQUIRE(rg.LastPoint() == point(180,90));

	rg.ScanningMode(kTopLeft);

	BOOST_REQUIRE(rg.FirstPoint() == point(-180,90));
	BOOST_REQUIRE(rg.LastPoint() == point(180,-90));

}

BOOST_AUTO_TEST_CASE(EQUALITY)
{
	reduced_gaussian_grid rg1;
	rg1.N(100);

	reduced_gaussian_grid rg2;
	rg2.N(101);
	
	BOOST_REQUIRE(rg1 != rg2);

	rg2.N(100);
	rg1.Nj(200);
	rg2.Nj(200);

	BOOST_REQUIRE(rg1 == rg2);

	rg1.BottomLeft(point(25,60));
	rg2.BottomLeft(point(25,60));
	rg1.TopRight(point(30,70));
	rg2.TopRight(point(30,71));

	BOOST_REQUIRE(rg1 != rg2);

	rg2.TopRight(point(30,70));
	rg1.N(4);
	rg2.N(4);

	std::vector<int> lons({5, 10, 15, 15, 15, 15, 10, 5});
	rg1.NumberOfLongitudesAlongParallels(lons);
	rg2.NumberOfLongitudesAlongParallels(lons);

	BOOST_REQUIRE(rg1 == rg2);

	lons = std::vector<int>({5, 10, 15, 16, 16, 15, 10, 5});
	rg2.NumberOfLongitudesAlongParallels(lons);

	BOOST_REQUIRE(rg1 != rg2);

}

BOOST_AUTO_TEST_CASE(LATLON)
{
	reduced_gaussian_grid rg;

	rg.ScanningMode(kTopLeft);
	rg.BottomLeft(point(25,68));
	rg.TopRight(point(33,75));

	rg.N(4);
	rg.Nj(8);
	std::vector<int> lons({5, 10, 15, 15, 15, 15, 10, 5});
	rg.NumberOfLongitudesAlongParallels(lons);

	BOOST_REQUIRE(rg.LatLon(0) == point(25,75));
	BOOST_REQUIRE(rg.LatLon(1) == point(27,75));
	BOOST_REQUIRE(rg.LatLon(2) == point(29,75));
	BOOST_REQUIRE(rg.LatLon(4) == point(33,75));

	BOOST_REQUIRE(rg.LatLon(5) == point(25,74));

	auto pt = rg.LatLon(6);
	BOOST_CHECK_CLOSE(pt.X(), 25.8889, kEpsilon);
	BOOST_REQUIRE(pt.Y() == 74);

	BOOST_REQUIRE(rg.LatLon(15) == point(25,73));

	pt = rg.LatLon(16);
	BOOST_CHECK_CLOSE(pt.X(), 25.5714, kEpsilon);
	BOOST_REQUIRE(pt.Y() == 73);

	BOOST_REQUIRE(rg.LatLon(29) == point(33,73));

	BOOST_REQUIRE(rg.LatLon(75) == point(25,69));
	pt = rg.LatLon(76);
	BOOST_CHECK_CLOSE(pt.X(), 25.8889, kEpsilon);
	BOOST_REQUIRE(pt.Y() == 69);
}

BOOST_AUTO_TEST_CASE(LATLON_PACIFIC)
{
	reduced_gaussian_grid rg;

	rg.ScanningMode(kTopLeft);
	rg.BottomLeft(point(0,0));
	rg.TopRight(point(357.6,90));

	rg.N(4);
	rg.Nj(8);
	std::vector<int> lons({50, 100, 150, 150, 150, 150, 100, 50});
	rg.NumberOfLongitudesAlongParallels(lons);

	BOOST_REQUIRE(rg.LatLon(0) == point(0,90));

	auto pt = rg.LatLon(3);
	BOOST_CHECK_CLOSE(pt.X(), 21.89387, kEpsilon);
	BOOST_REQUIRE(pt.Y() == 90);

	pt = rg.LatLon(41); // X = 299.21
	BOOST_CHECK_CLOSE(pt.X(), 299.2163, kEpsilon);
	BOOST_REQUIRE(pt.Y() == 90);

}



#if 0
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

#endif
