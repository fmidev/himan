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
vector<vector<double>> heights_all;

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

void init()
{

	if (values_all.size() > 0)
	{
		return;
	}
	
	// HEIGHTS
	
	// Level 4, closest to ground, in meters
	
	vector<double> heights = {
		15., 13., 19.,
		14., 12., 31.,
		16., 5.,  22.
	};

	heights_all.push_back(heights);
	
	// Level 3
	
	heights = {
		27., 33., 29.,
		24., 42., 41.,
		29., 17., 32.
	};
	
	heights_all.push_back(heights);

	// Level 2
	
	heights = {
		107., 167., 130.,
		kFloatMissing, 152., 170.,
		139., 120., 151.
	};
	
	heights_all.push_back(heights);

	// Level 1
	
	heights = {
		402., 280., 133.,
		464., 412., 661.,
		466., 119., kFloatMissing
	};
	
	heights_all.push_back(heights);

	// VALUES
	
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
	assert(heights_all.size() == 4);
	
	cout << "=== VALUES ===\n";
	dump(values_all);
	cout << "===HEIGHTS ===\n";
	dump(heights_all);
	cout << endl;

}

BOOST_AUTO_TEST_CASE(MODIFIER_MIN)
{

	init();

	auto mod = unique_ptr<modifier_min> (new modifier_min);

	for (size_t i = 0; i < level_count; i++)
	{
		mod->Process(values_all[i], heights_all[i]);
	}

	auto result = mod->Result();

	//vdump(result);
	BOOST_REQUIRE(result[0] == -26);
	BOOST_REQUIRE(result[2] == 7);
	BOOST_REQUIRE(result[7] == 5);
}

BOOST_AUTO_TEST_CASE(MODIFIER_MAX)
{

	init();

	auto mod = unique_ptr<modifier_max> (new modifier_max);

	for (size_t i = 0; i < level_count; i++)
	{
		mod->Process(values_all[i], heights_all[i]);
	}

	auto result = mod->Result();

	//vdump(result);
	BOOST_REQUIRE(result[2] == 7);
	BOOST_REQUIRE(result[3] == -8);
	BOOST_REQUIRE(result[7] == 6);

}

BOOST_AUTO_TEST_CASE(MODIFIER_MAXMIN)
{

	init();

	auto mod = unique_ptr<modifier_maxmin> (new modifier_maxmin);

	for (size_t i = 0; i < level_count; i++)
	{
		mod->Process(values_all[i], heights_all[i]);
	}

	auto result = mod->Result();

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

	auto mod = unique_ptr<modifier_sum> (new modifier_sum);

	for (size_t i = 0; i < level_count; i++)
	{
		mod->Process(values_all[i], heights_all[i]);
	}

	auto result = mod->Result();

	//vdump(result);
	BOOST_REQUIRE(result[0] == -69);
	BOOST_REQUIRE(result[2] == 14);
	BOOST_REQUIRE(result[8] == -1);
}

BOOST_AUTO_TEST_CASE(MODIFIER_MEAN)
{

	init();

	auto mod = unique_ptr<modifier_mean> (new modifier_mean);

	for (size_t i = 0; i < level_count; i++)
	{
		mod->Process(values_all[i], heights_all[i]);
	}

	auto result = mod->Result();

	//vdump(result);
	BOOST_CHECK_CLOSE(result[0], -22.9134, kEpsilon);
	BOOST_REQUIRE(result[2] == 7);
	BOOST_CHECK_CLOSE(result[8], -1.38372, kEpsilon);

}

BOOST_AUTO_TEST_CASE(MODIFIER_COUNT)
{

	init();

	auto mod = unique_ptr<modifier_count> (new modifier_count);

	vector<double> findv = {
		-5, 21, 7,
		-24, 3, 15,
		-3, kFloatMissing, 116
	};

	//vdump(findv);

	mod->FindValue(findv);

	for (size_t i = 0; i < level_count; i++)
	{
		mod->Process(values_all[i], heights_all[i]);
	}

	auto values = mod->Result();

	// vdump(values);
	BOOST_REQUIRE(values[0] == 1);
	BOOST_REQUIRE(values[1] == 0);
	BOOST_REQUIRE(values[6] == 2);
	BOOST_REQUIRE(values[7] == 0);
	
}

BOOST_AUTO_TEST_CASE(MODIFIER_FINDHEIGHT)
{

	init();

	auto mod = unique_ptr<modifier_findheight> (new modifier_findheight);

	vector<double> findv = {
		-23, 15, 0,
		-20, 3, 10,
		-3, 0, 116
	};

	//vdump(findv);

	mod->FindValue(findv);

	for (size_t i = 0; i < level_count; i++)
	{
		mod->Process(values_all[i], heights_all[i]);
	}

	auto values = mod->Result();

	//vdump(values);
	BOOST_REQUIRE(values[0] == 107);
	BOOST_REQUIRE(values[2] == kFloatMissing);
	BOOST_CHECK_CLOSE(values[3], 200, kEpsilon);
	BOOST_CHECK_CLOSE(values[6], 18.8888, kEpsilon);

	mod->FindNth(2);
	mod->Clear();

	for (size_t i = 0; i < level_count; i++)
	{
		mod->Process(values_all[i], heights_all[i]);
	}

	values = mod->Result();

	//vdump(values);

	BOOST_REQUIRE(values[0] == kFloatMissing);
	BOOST_CHECK_CLOSE(values[5], 66.8, kEpsilon);
	BOOST_CHECK_CLOSE(values[6], 443.448, kEpsilon);

	mod->FindNth(0);
	mod->Clear();

	for (size_t i = 0; i < level_count; i++)
	{
		mod->Process(values_all[i], heights_all[i]);
	}

	values = mod->Result();

	//vdump(values);

	BOOST_REQUIRE(values[0] == 107);
	BOOST_REQUIRE(values[2] == kFloatMissing);
	BOOST_CHECK_CLOSE(values[5], 321.077, kEpsilon);

}

BOOST_AUTO_TEST_CASE(MODIFIER_FINDVALUE)
{

	init();

	auto mod = unique_ptr<modifier_findvalue> (new modifier_findvalue);

	vector<double> findh = {
		50, 0, 51,
		52, 3, 15, 
		103, 0, 416, 
	};

//	vdump(findh);

	mod->FindValue(findh);

	for (size_t i = 0; i < level_count; i++)
	{
		mod->Process(values_all[i], heights_all[i]);
	}

	auto values = mod->Result();

//	vdump(values);
	BOOST_REQUIRE(values[0] == -17.3);
	BOOST_REQUIRE(values[1] == 15); // clamp
	BOOST_REQUIRE(values[2] == 7);
	BOOST_REQUIRE(values[8] == kFloatMissing);

}
