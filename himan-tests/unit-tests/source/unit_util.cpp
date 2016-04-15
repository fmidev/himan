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

BOOST_AUTO_TEST_CASE(CENTRAL_DIFFERENCE)
{
    // Filter a plane with given filter kernel
    // Declare matrices
    himan::matrix<double> A(5,5,1,kFloatMissing);
    himan::matrix<double> B;
    himan::matrix<double> C;

	// Matrices containing the correct solution
	himan::matrix<double> D(5,5,1,kFloatMissing);
	himan::matrix<double> E(5,5,1,kFloatMissing);

    // Fill matrix A and solution matrices D and E
    for(size_t i=0; i < A.Size(); ++i)
    {
        A.Set(i, double(i));
		D.Set(i,1.0/double(1+i/5));
		E.Set(i,5.0);
    }

	std::pair<himan::matrix<double>,himan::matrix<double>> grad_A;

	// Declare vectors for grid spacing
	std::vector<double> dx {1.0,2.0,3.0,4.0,5.0};
	std::vector<double> dy(5,1.0);

	grad_A = himan::util::CentralDifference(A,dx,dy);
	B = std::get<0>(grad_A);
	C = std::get<1>(grad_A);

	BOOST_CHECK(B==D && C==E);
	
} 

BOOST_AUTO_TEST_CASE(MAKESQLINTERVAL)
{

	forecast_time f1("2015-01-09 00:00:00", "2015-01-09 12:00:00");

	BOOST_REQUIRE(util::MakeSQLInterval(f1) == "12:00:00");

	f1.StepResolution(kMinuteResolution);

	BOOST_REQUIRE(util::MakeSQLInterval(f1) == "12:00:00");

	forecast_time f2("2015-01-09 00:00:00", "2015-01-09 00:15:00");

	BOOST_REQUIRE(util::MakeSQLInterval(f2) == "00:00:00");

	f2.StepResolution(kMinuteResolution);	

	BOOST_REQUIRE(util::MakeSQLInterval(f2) == "00:15:00");

	forecast_time f3("2015-01-09 00:00:00", "2015-01-19 00:16:00");

	BOOST_REQUIRE(util::MakeSQLInterval(f3) == "240:00:00");

	forecast_time f4("2015-01-09 00:00:00", "2015-10-19 00:00:00");

	BOOST_REQUIRE(util::MakeSQLInterval(f4) == "6792:00:00");

	forecast_time f5("2015-01-09 00:00:00", "2015-01-09 00:00:00");

	BOOST_REQUIRE(util::MakeSQLInterval(f5) == "00:00:00");
}

BOOST_AUTO_TEST_CASE(EXPAND)
{
	setenv("BOOST_TEST", "xyz", 1);

	string test = "$BOOST_TEST/asdf";

	string expanded = util::Expand(test);

	BOOST_REQUIRE(expanded == "xyz/asdf");

	setenv("BOOST_TEST_2", "123", 1);

	test = "$BOOST_TEST/asdf/$BOOST_TEST_2";

	expanded = util::Expand(test);

	BOOST_REQUIRE(expanded == "xyz/asdf/123");

}
