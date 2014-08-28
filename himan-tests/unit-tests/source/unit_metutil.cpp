#include "himan_unit.h"
#include "metutil.h"

#define BOOST_TEST_MODULE metutil

using namespace std;
using namespace himan;

const double kEpsilon = 1e-3;

BOOST_AUTO_TEST_CASE(LCL_SLOW)
{

	// "slow" calculation of LCL

	lcl_t LCL = himan::metutil::LCL_(85000,273.15 + 16.5828, 273.15 + -1.45402);

	BOOST_CHECK_CLOSE(LCL.P, 64829.6, kEpsilon);
	BOOST_CHECK_CLOSE(LCL.T, 273.15 + -5.01719, kEpsilon);
}

BOOST_AUTO_TEST_CASE(LCL_FAST)
{

        // fast calculation of LCL

	lcl_t LCL = himan::metutil::LCL_(85000,273.15 + 16.6453, 273.15 + -1.29777);

	BOOST_CHECK_CLOSE(LCL.P, 64878.5986, kEpsilon);
	BOOST_CHECK_CLOSE(LCL.T, 273.15 + -4.90058, kEpsilon);
}

BOOST_AUTO_TEST_CASE(LCL_SMARTTOOL_COMPATIBILITY)
{
	// testing if himan LCL is compatible with smarttool LCL

	double T = 16.5828;
	double Td = -1.45402;
	double P = 850;

	lcl_t LCL = himan::metutil::LCL_(P*100, 273.15 + T, 273.15 + Td);

	BOOST_CHECK_CLOSE(LCL.P, 100*647.3825, 1);

}

BOOST_AUTO_TEST_CASE(MIXING_RATIO)
{
	double TD = 4.2;
	double P = 850;

	double ratio = metutil::MixingRatio_(273.15 + TD, 100 * P);

	BOOST_CHECK_CLOSE(ratio, 6.0955, kEpsilon);
}

BOOST_AUTO_TEST_CASE(SATURATED_MIXING_RATIO)
{

	// saturated mixing ratio

	double ratio = metutil::MixingRatio_(290, 98000);

	BOOST_CHECK_CLOSE(ratio, 12.4395, kEpsilon);

}

BOOST_AUTO_TEST_CASE(ES)
{

	// water vapour pressure

	double E = metutil::Es_(285);

	BOOST_CHECK_CLOSE(E, 1389.859, kEpsilon);

	// negative temperatures

	E = metutil::Es_(266);

	BOOST_CHECK_CLOSE(E, 333.356, kEpsilon);

}

BOOST_AUTO_TEST_CASE(DEWPOINT_HIGH_RH)
{

	double TD = metutil::DewPointFromHighRH_(292.5, 92.1);

	BOOST_CHECK_CLOSE(TD, 290.9200, kEpsilon);

	// negative temperatures

	TD = metutil::DewPointFromHighRH_(264, 57.44);

	BOOST_CHECK_CLOSE(TD, 255.488, kEpsilon);

}

BOOST_AUTO_TEST_CASE(DEWPOINT_LOW_RH)
{

	double TD = metutil::DewPointFromLowRH_(292.5, 22);

	BOOST_CHECK_CLOSE(TD, 292.3959, kEpsilon);

	// negative temperatures

	TD = metutil::DewPointFromLowRH_(264, 42.0);

	BOOST_CHECK_CLOSE(TD, 263.9865, kEpsilon);

}

