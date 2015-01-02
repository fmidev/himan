//#define BOOST_TEST_DYN_LINK

#include "himan_unit.h"
#include "modifier.h"

#define BOOST_TEST_MODULE modifier

using namespace std;
using namespace himan;

const double kEpsilon = 1e-3;
const size_t arr_size = 9;
const size_t level_count = 4;

vector<vector<double>> values_all;
vector<vector<double>> heights_all_meters;
vector<vector<double>> heights_all_pascals; // hpa actually

void vdump(const vector<double>& vec)
{
	int j = 0;

	for (size_t i = 0; i < vec.size(); i++, j++)
	{
		
		cout << " " << vec[i];
		if (j >= 2)
		{
			cout << endl;
			j = -1;
		}			
	}
}

void dump(const vector<vector<double>>& vec)
{
	
	for (size_t i = 0; i < level_count; i++)
	{
		cout << "=== Level: " << level_count-i <<  " ===" << endl;
		vdump(vec[i]);
	}
}

void init_values()
{
	// Level 4, closest to ground
	
	vector<double> values = {
		-5., 15., kFloatMissing,
		-8., 2., 1.,
		-1., 5,  2.
	};

	values_all.push_back(values);
	
	// Level 3
	
	values = {
		-15., 18., 7,
		-18., 11., 11.,
		-10., 6,  -2.
	};
	
	values_all.push_back(values);

	// Level 2
	
	values = {
		-23., 19., kFloatMissing,
		-28., 16., 6.,
		-30., 6,  -1.
	};
	
	values_all.push_back(values);

	// Level 1
	
	values = {
		-26., 20., 7.,
		-23., 16., 19.,
		-1., kFloatMissing,  -10.
	};

	values_all.push_back(values);
	
	assert(values_all.size() == 4);

	cout << "=== VALUES ===\n";
	dump(values_all);
	cout << endl;
}

void init_heights_in_pascals()
{
	// Level 4, closest to ground, in hectopascals
	
	vector<double> heights = {
		995., 1000., 991.,
		1023., 1023., 1025.,
		994., 990.,  1000.
	};

	heights_all_pascals.push_back(heights);
	
	// Level 3
	
	heights = {
		920., 930, 915.,
		940., 1000., 1000,
		901., 899., 950.
	};
	
	heights_all_pascals.push_back(heights);

	// Level 2
	
	heights = {
		850., 850, 910.,
		kFloatMissing, 900., 920.,
		880., 800., 830.
	};
	
	heights_all_pascals.push_back(heights);

	// Level 1
	
	heights = {
		700., 700., 710.,
		730., 730., 500.,
		680., 600., kFloatMissing
	};
	
	heights_all_pascals.push_back(heights);


	assert(heights_all_pascals.size() == 4);
	
	
	cout << "=== HEIGHTS IN HECTOPASCALS ===\n";
	dump(heights_all_pascals);
	cout << endl;
}

void init_heights_in_meters()
{

	// Level 4, closest to ground, in meters
	
	vector<double> heights = {
		15., 13., 19.,
		14., 12., 31.,
		16., 5.,  22.
	};

	heights_all_meters.push_back(heights);
	
	// Level 3
	
	heights = {
		27., 33., 29.,
		24., 42., 41.,
		29., 17., 32.
	};
	
	heights_all_meters.push_back(heights);

	// Level 2
	
	heights = {
		107., 167., 130.,
		kFloatMissing, 152., 170.,
		139., 120., 151.
	};
	
	heights_all_meters.push_back(heights);

	// Level 1
	
	heights = {
		402., 280., 133.,
		464., 412., 661.,
		466., 119., kFloatMissing
	};
	
	heights_all_meters.push_back(heights);


	assert(heights_all_meters.size() == 4);
	
	
	cout << "=== HEIGHTS IN METERS ===\n";
	dump(heights_all_meters);
	cout << endl;
}

void init()
{

	if (values_all.size() > 0)
	{
		return;
	}

	init_values();
	init_heights_in_meters();
	init_heights_in_pascals();
}

BOOST_AUTO_TEST_CASE(MODIFIER_MIN)
{

	init();

	modifier_min mod;

	for (size_t i = 0; i < level_count; i++)
	{
		mod.Process(values_all[i], heights_all_meters[i]);
	}

	auto result = mod.Result();

	//vdump(result);
	BOOST_REQUIRE(result[0] == -26);
	BOOST_REQUIRE(result[2] == 7);
	BOOST_REQUIRE(result[7] == 5);
}

BOOST_AUTO_TEST_CASE(MODIFIER_MIN_PA)
{

	init();

	modifier_min mod;

	mod.HeightInMeters(false);

	for (size_t i = 0; i < level_count; i++)
	{
		mod.Process(values_all[i], heights_all_pascals[i]);
	}

	auto result = mod.Result();

	//vdump(result);
	BOOST_REQUIRE(result[0] == -26);
	BOOST_REQUIRE(result[2] == 7);
	BOOST_REQUIRE(result[7] == 5);
}

BOOST_AUTO_TEST_CASE(MODIFIER_MAX)
{

	init();

	modifier_max mod;

	for (size_t i = 0; i < level_count; i++)
	{
		mod.Process(values_all[i], heights_all_meters[i]);
	}

	auto result = mod.Result();

	//vdump(result);
	BOOST_REQUIRE(result[2] == 7);
	BOOST_REQUIRE(result[3] == -8);
	BOOST_REQUIRE(result[7] == 6);

}

BOOST_AUTO_TEST_CASE(MODIFIER_MAX_PA)
{

	init();

	modifier_max mod;
	mod.HeightInMeters(false);

	for (size_t i = 0; i < level_count; i++)
	{
		mod.Process(values_all[i], heights_all_pascals[i]);
	}

	auto result = mod.Result();

	//vdump(result);
	BOOST_REQUIRE(result[2] == 7);
	BOOST_REQUIRE(result[3] == -8);
	BOOST_REQUIRE(result[7] == 6);

}

BOOST_AUTO_TEST_CASE(MODIFIER_MAXMIN)
{

	init();

	modifier_maxmin mod;

	for (size_t i = 0; i < level_count; i++)
	{
		mod.Process(values_all[i], heights_all_meters[i]);
	}

	auto result = mod.Result();

	//vdump(result);
	// mins
	BOOST_REQUIRE(result[0] == -26);
	BOOST_REQUIRE(result[2] == 7);
	BOOST_REQUIRE(result[7] == 5);
	// maxs
	BOOST_REQUIRE(result[arr_size + 2] == 7);
	BOOST_REQUIRE(result[arr_size + 3] == -8);
	BOOST_REQUIRE(result[arr_size + 7] == 6);
}

BOOST_AUTO_TEST_CASE(MODIFIER_MAXMIN_PA)
{

	init();

	modifier_maxmin mod;
	mod.HeightInMeters(false);

	for (size_t i = 0; i < level_count; i++)
	{
		mod.Process(values_all[i], heights_all_pascals[i]);
	}

	auto result = mod.Result();

	//vdump(result);
	// mins
	BOOST_REQUIRE(result[0] == -26);
	BOOST_REQUIRE(result[2] == 7);
	BOOST_REQUIRE(result[7] == 5);
	// maxs
	BOOST_REQUIRE(result[arr_size + 2] == 7);
	BOOST_REQUIRE(result[arr_size + 3] == -8);
	BOOST_REQUIRE(result[arr_size + 7] == 6);
}

BOOST_AUTO_TEST_CASE(MODIFIER_SUM)
{

	init();

	modifier_sum mod;

	for (size_t i = 0; i < level_count; i++)
	{
		mod.Process(values_all[i], heights_all_meters[i]);
	}

	auto result = mod.Result();

	//vdump(result);
	BOOST_REQUIRE(result[0] == -69);
	BOOST_REQUIRE(result[2] == 14);
	BOOST_REQUIRE(result[8] == -1);
}

BOOST_AUTO_TEST_CASE(MODIFIER_SUM_PA)
{

	init();

	modifier_sum mod;
	mod.HeightInMeters(false);

	for (size_t i = 0; i < level_count; i++)
	{
		mod.Process(values_all[i], heights_all_pascals[i]);
	}

	auto result = mod.Result();

	//vdump(result);
	BOOST_REQUIRE(result[0] == -69);
	BOOST_REQUIRE(result[2] == 14);
	BOOST_REQUIRE(result[8] == -1);
}

BOOST_AUTO_TEST_CASE(MODIFIER_MEAN)
{

	init();

	modifier_mean mod;

	for (size_t i = 0; i < level_count; i++)
	{
		mod.Process(values_all[i], heights_all_meters[i]);
	}

	auto result = mod.Result();

	//vdump(result);
	BOOST_CHECK_CLOSE(result[0], -22.9134, kEpsilon);
	BOOST_REQUIRE(result[2] == 7);
	BOOST_CHECK_CLOSE(result[8], -1.38372, kEpsilon);

}

BOOST_AUTO_TEST_CASE(MODIFIER_MEAN_USING_CALCULATION_FINISHED)
{

	init();

	modifier_mean mod;

	for (size_t i = 0; i < level_count && !mod.CalculationFinished(); i++)
	{
		mod.Process(values_all[i], heights_all_meters[i]);
	}

	auto result = mod.Result();

	//vdump(result);
	BOOST_CHECK_CLOSE(result[0], -22.9134, kEpsilon);
	BOOST_REQUIRE(result[2] == 7);
	BOOST_CHECK_CLOSE(result[8], -1.38372, kEpsilon);

}

BOOST_AUTO_TEST_CASE(MODIFIER_MEAN_PA)
{

	init();

	modifier_mean mod;
	mod.HeightInMeters(false);

	for (size_t i = 0; i < level_count; i++)
	{
		mod.Process(values_all[i], heights_all_pascals[i]);
	}

	auto result = mod.Result();

	//vdump(result);
	BOOST_CHECK_CLOSE(result[0], -19.50847, kEpsilon);
	BOOST_REQUIRE(result[2] == 7);
	BOOST_CHECK_CLOSE(result[8], -1.05882, kEpsilon);

}

BOOST_AUTO_TEST_CASE(MODIFIER_COUNT)
{

	init();

	modifier_count mod;

	vector<double> findv = {
		-5, 21, 7,
		-24, 3, 15,
		-3, kFloatMissing, 116
	};

	//vdump(findv);

	mod.FindValue(findv);

	for (size_t i = 0; i < level_count; i++)
	{
		mod.Process(values_all[i], heights_all_meters[i]);
	}

	auto values = mod.Result();

	// vdump(values);
	BOOST_REQUIRE(values[0] == 1);
	BOOST_REQUIRE(values[1] == 0);
	BOOST_REQUIRE(values[6] == 2);
	BOOST_REQUIRE(values[7] == 0);
	
}

BOOST_AUTO_TEST_CASE(MODIFIER_COUNT_PA)
{

	init();

	modifier_count mod;
	mod.HeightInMeters(false);

	vector<double> findv = {
		-5, 21, 7,
		-24, 3, 15,
		-3, kFloatMissing, 116
	};

	//vdump(findv);

	mod.FindValue(findv);

	for (size_t i = 0; i < level_count; i++)
	{
		mod.Process(values_all[i], heights_all_pascals[i]);
	}

	auto values = mod.Result();

	//vdump(values);
	BOOST_REQUIRE(values[0] == 1);
	BOOST_REQUIRE(values[1] == 0);
	BOOST_REQUIRE(values[6] == 2);
	BOOST_REQUIRE(values[7] == 0);
	
}

BOOST_AUTO_TEST_CASE(MODIFIER_FINDHEIGHT)
{

	init();

	modifier_findheight mod;

	vector<double> findv = {
		-23, 15, 0,
		-20, 3, 10,
		-3, 0, 116
	};

	//vdump(findv);

	mod.FindValue(findv);

	for (size_t i = 0; i < level_count; i++)
	{
		mod.Process(values_all[i], heights_all_meters[i]);
	}

	auto values = mod.Result();

	//vdump(values);
	BOOST_REQUIRE(values[0] == 107);
	BOOST_REQUIRE(values[2] == kFloatMissing);
	BOOST_CHECK_CLOSE(values[3], 200, kEpsilon);
	BOOST_CHECK_CLOSE(values[6], 18.8888, kEpsilon);

	mod.FindNth(2);
	mod.Clear();

	for (size_t i = 0; i < level_count; i++)
	{
		mod.Process(values_all[i], heights_all_meters[i]);
	}

	values = mod.Result();

	//vdump(values);

	BOOST_REQUIRE(values[0] == kFloatMissing);
	BOOST_CHECK_CLOSE(values[5], 66.8, kEpsilon);
	BOOST_CHECK_CLOSE(values[6], 443.448, kEpsilon);

	mod.FindNth(0);
	mod.Clear();

	for (size_t i = 0; i < level_count; i++)
	{
		mod.Process(values_all[i], heights_all_meters[i]);
	}

	values = mod.Result();

	//vdump(values);

	BOOST_REQUIRE(values[0] == 107);
	BOOST_REQUIRE(values[2] == kFloatMissing);
	BOOST_CHECK_CLOSE(values[5], 321.077, kEpsilon);

}

BOOST_AUTO_TEST_CASE(MODIFIER_FINDHEIGHT_PA)
{

	init();

	modifier_findheight mod;
	mod.HeightInMeters(false);

	vector<double> findv = {
		-23, 15, 0,
		-20, 3, 10,
		-3, 0, 116
	};

	//vdump(findv);

	mod.FindValue(findv);

	for (size_t i = 0; i < level_count; i++)
	{
		mod.Process(values_all[i], heights_all_pascals[i]);
	}

	auto values = mod.Result();

	//vdump(values);
	BOOST_REQUIRE(values[0] == 850);
	BOOST_REQUIRE(values[2] == kFloatMissing);
	BOOST_CHECK_CLOSE(values[3], 856, kEpsilon);
	BOOST_CHECK_CLOSE(values[6], 973.33333, kEpsilon);

	mod.FindNth(2);
	mod.Clear();

	for (size_t i = 0; i < level_count; i++)
	{
		mod.Process(values_all[i], heights_all_pascals[i]);
	}

	values = mod.Result();

	//vdump(values);

	BOOST_REQUIRE(values[0] == kFloatMissing);
	BOOST_CHECK_CLOSE(values[5], 984, kEpsilon);
	BOOST_CHECK_CLOSE(values[6], 693.79310, kEpsilon);

	mod.FindNth(0);
	mod.Clear();

	for (size_t i = 0; i < level_count; i++)
	{
		mod.Process(values_all[i], heights_all_pascals[i]);
	}

	values = mod.Result();

	//vdump(values);

	BOOST_REQUIRE(values[0] == 850);
	BOOST_REQUIRE(values[2] == kFloatMissing);
	BOOST_CHECK_CLOSE(values[5], 790.76923, kEpsilon);

}

BOOST_AUTO_TEST_CASE(MODIFIER_FINDVALUE)
{

	init();

	modifier_findvalue mod;

	vector<double> findh = {
		50, 0, 51,
		52, 3, 15, 
		103, 0, 416, 
	};

//	vdump(findh);

	mod.FindValue(findh);

	for (size_t i = 0; i < level_count; i++)
	{
		mod.Process(values_all[i], heights_all_meters[i]);
	}

	auto values = mod.Result();

	//vdump(values);
	BOOST_REQUIRE(values[0] == -17.3);
	BOOST_REQUIRE(values[1] == 15); // clamp
	BOOST_REQUIRE(values[2] == 7);
	BOOST_REQUIRE(values[8] == kFloatMissing);

}

BOOST_AUTO_TEST_CASE(MODIFIER_FINDVALUE_PA)
{

	init();

	modifier_findvalue mod;
	mod.HeightInMeters(false);

	vector<double> findh = {
		995, 1010, 700,
		852, 733, 850, 
		690, 899, 975, 
	};

//	vdump(findh);

	mod.FindValue(findh);

	for (size_t i = 0; i < level_count; i++)
	{
		mod.Process(values_all[i], heights_all_pascals[i]);
	}

	auto values = mod.Result();

	//vdump(values);
	BOOST_REQUIRE(values[0] == -5);
	BOOST_REQUIRE(values[1] == kFloatMissing); 
	BOOST_CHECK_CLOSE(values[3], -20.0952, kEpsilon);
	BOOST_REQUIRE(values[8] == 0);

}