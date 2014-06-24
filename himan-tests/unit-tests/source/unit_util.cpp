#include "himan_unit.h"
#include "util.h"

#define BOOST_TEST_MODULE util

using namespace std;
using namespace himan;

const double kEpsilon = 1e-3;

BOOST_AUTO_TEST_CASE(UV_TO_GEOGRAPHICAL)
{
	// Transform grid coordinates to lat and lon in stereographic projection

	himan::point stereoUV(8.484046, 3.804569);
	double lon = 72.79;

	himan::point latlon = util::UVToGeographical(lon, stereoUV);

	BOOST_CHECK_CLOSE(latlon.X(), 6.144442, kEpsilon); 
	BOOST_CHECK_CLOSE(latlon.Y(), -6.978511, kEpsilon); 

	stereoUV.X(-0.2453410);
	stereoUV.Y(0.5808838);
	lon = 23.39;

	latlon = util::UVToGeographical(lon, stereoUV);

	BOOST_CHECK_CLOSE(latlon.X(), 5.4238806e-03, kEpsilon); 
	BOOST_CHECK_CLOSE(latlon.Y(), 0.6305464, kEpsilon); 

}
