#include "himan_unit.h"
#include "util.h"

#define BOOST_TEST_MODULE util

using namespace std;
using namespace himan;

const double kEpsilon = 1e-3;

BOOST_AUTO_TEST_CASE(LCL_SLOW)
{

        // "slow" calculation of LCL

	std::vector<double> LCL = himan::util::LCL(85000,273.15 + 16.5828, 273.15 + -1.45402);

	BOOST_CHECK_CLOSE(LCL[0], 64829.6, kEpsilon);
	BOOST_CHECK_CLOSE(LCL[1], 273.15 + -5.01719, kEpsilon);
}

BOOST_AUTO_TEST_CASE(LCL_FAST)
{

        // fast calculation of LCL

	std::vector<double> LCL = himan::util::LCL(85000,273.15 + 16.6453, 273.15 + -1.29777);

	BOOST_CHECK_CLOSE(LCL[0], 64878.5986, kEpsilon);
	BOOST_CHECK_CLOSE(LCL[1], 273.15 + -4.90058, kEpsilon);
}

BOOST_AUTO_TEST_CASE(SATURATED_MIXING_RATIO)
{

	// saturated mixing ratio

	double r = util::MixingRatio(290, 98000);

	BOOST_CHECK_CLOSE(r, 12.4395, kEpsilon);

}

BOOST_AUTO_TEST_CASE(ES)
{

	// water vapour pressure

	double E = util::Es(285);

	BOOST_CHECK_CLOSE(E, 1389.859, kEpsilon);

	// negative temperatures

	E = util::Es(266);

	BOOST_CHECK_CLOSE(E, 333.356, kEpsilon);

}
