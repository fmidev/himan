#include "himan_unit.h"
#include "util.h"

#define BOOST_TEST_MODULE util

using namespace std;
using namespace himan;

const double kEpsilon = 1e-3;

BOOST_AUTO_TEST_CASE(LCL_SLOW)
{

        // "slow" calculation of LCL

	std::vector<double> LCL = himan::util::LCL(850,16.5828,-1.45402);

	BOOST_CHECK_CLOSE(LCL[0], 648.296, kEpsilon);
	BOOST_CHECK_CLOSE(LCL[1], -5.01719, kEpsilon);
}

BOOST_AUTO_TEST_CASE(LCL_FAST)
{

        // fast calculation of LCL

	std::vector<double> LCL = himan::util::LCL(850,16.6453,-1.29777);

	BOOST_CHECK_CLOSE(LCL[0], 648.793, kEpsilon);
	BOOST_CHECK_CLOSE(LCL[1], -4.90058, kEpsilon);
}
