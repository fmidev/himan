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

BOOST_AUTO_TEST_CASE(ES2)
{
	// water vapour pressure, another formula

	double E = metutil::Es2_(285);

	BOOST_CHECK_CLOSE(E, 1388.65877, kEpsilon);

	// negative temperatures

	E = metutil::Es2_(266);

	BOOST_CHECK_CLOSE(E, 357.80213, kEpsilon);



}

BOOST_AUTO_TEST_CASE(DEWPOINT_SIMPLE)
{

	double TD = metutil::DewPointFromRHSimple_(292.5, 92.1);

	BOOST_CHECK_CLOSE(TD, 290.9200, kEpsilon);

	// negative temperatures

	TD = metutil::DewPointFromRHSimple_(264, 57.44);

	BOOST_CHECK_CLOSE(TD, 255.488, kEpsilon);

}

BOOST_AUTO_TEST_CASE(DEWPOINT)
{

	double TD = metutil::DewPointFromRH_(292.5, 22);

	BOOST_CHECK_CLOSE(TD, 270.3936, kEpsilon);

	// negative temperatures

	TD = metutil::DewPointFromRH_(264, 42.0);

	BOOST_CHECK_CLOSE(TD, 253.2915, kEpsilon);

}

BOOST_AUTO_TEST_CASE(THETA)
{
	double tpot = metutil::Theta_(19+273.15, 94000);

	BOOST_CHECK_CLOSE(tpot, 297.36600, kEpsilon);

	tpot = metutil::Theta_(-20+273.15, 83400);

	BOOST_CHECK_CLOSE(tpot, 266.63269, kEpsilon);

}

BOOST_AUTO_TEST_CASE(THETAE)
{
	// These test values are taken from Boltons paper

	// The somewhat large differences 0.1 ... 0.5 to reference values
	// are most likely caused by mixing ratio formula which is not the
	// same one Bolton is using.

	double T = 273.15 + 30;
	double P = 100 * 1000;

	double thetae = metutil::ThetaE_(T,P);

	BOOST_CHECK_CLOSE(thetae, 386.28, 0.1);

	T -= 10;

	thetae = metutil::ThetaE_(T,P);	

	BOOST_CHECK_CLOSE(thetae, 335.61, 0.1);

	T -= 20;

	thetae = metutil::ThetaE_(T,P);	

	BOOST_CHECK_CLOSE(thetae, 283.60, 0.1);

	T -= 30;

	thetae = metutil::ThetaE_(T,P);	

	BOOST_CHECK_CLOSE(thetae, 244.01, 0.1);

	T = 273.15 + 20;
	P = 100 * 700;

	thetae = metutil::ThetaE_(T,P);	

	BOOST_CHECK_CLOSE(thetae, 394.71, 0.1);

	T -= 20;

	thetae = metutil::ThetaE_(T,P);	

	BOOST_CHECK_CLOSE(thetae, 319.13, 0.1);	

	T -= 30;

	thetae = metutil::ThetaE_(T,P);	

	BOOST_CHECK_CLOSE(thetae, 270.57, 0.2);

	T = 273.15 - 30;
	P = 100 * 200;

	thetae = metutil::ThetaE_(T,P);	

	BOOST_CHECK_CLOSE(thetae, 391.82, 0.5);	

	T -= 20;

	thetae = metutil::ThetaE_(T,P);	

	BOOST_CHECK_CLOSE(thetae, 354.11, 0.1);	

}

BOOST_AUTO_TEST_CASE(THETAW)
{

	// Reference data from smithsonian tables (319)

	double T = 12 + 273.15;
	double P = 36960;

	double thetaE = metutil::ThetaE_(T,P);
	double thetaW = metutil::ThetaW_(thetaE, P);

	BOOST_CHECK_CLOSE(thetaW - 273.15, 39.9, 0.1);

	T = -32 + 273.15;
	P = 18820;

	thetaE = metutil::ThetaE_(T,P);
	thetaW = metutil::ThetaW_(thetaE, P);

	BOOST_CHECK_CLOSE(thetaW - 273.15, 30.97, 0.1);

	T = 38 + 273.15;
	P = 106980;

	thetaE = metutil::ThetaE_(T,P);
	thetaW = metutil::ThetaW_(thetaE, P);

	BOOST_CHECK_CLOSE(thetaW - 273.15, 36.05, 0.1);

	T = 10 + 273.15;
	P = 90800;

	thetaE = metutil::ThetaE_(T,P);
	thetaW = metutil::ThetaW_(thetaE, P);

	BOOST_CHECK_CLOSE(thetaW - 273.15, 14.02, 0.1);

	T = -14 + 273.15;
	P = 64800;

	thetaE = metutil::ThetaE_(T,P);
	thetaW = metutil::ThetaW_(thetaE, P);

	BOOST_CHECK_CLOSE(thetaW - 273.15, 7.54, 0.1);

	T = -50 + 273.15;
	P = 30170;

	thetaE = metutil::ThetaE_(T,P);
	thetaW = metutil::ThetaW_(thetaE, P);

	BOOST_CHECK_CLOSE(thetaW - 273.15, 13.68, 0.1);
}

/*BOOST_AUTO_TEST_CASE(GAMMAW)
{
	double LR = metutil::Gammaw_(100000, -40 + constants::kKelvin);

	BOOST_CHECK_CLOSE(LR, 0.00211237, kEpsilon);
}
*/