//#define BOOST_TEST_DYN_LINK

#include "himan_unit.h"
#include "modifier.h"

#define BOOST_TEST_MODULE modifier

using namespace std;
using namespace himan;

const double kEpsilon = 1e-3;
const size_t arr_size = 20;
const size_t level_count = 4;

vector<vector<double>> values_all;
vector<vector<double>> heights_all;

void vdump(vector<double> vec)
{
	int j = 0;

	for (size_t i = 0; i < vec.size(); i++, j++)
	{
		
		cout << vec[i] << "\t" ;
		if (j > 3)
		{
			cout << endl;
			j = -1;
		}			
	}
}

void dump(vector<vector<double>> vec)
{
	
	for (size_t i = 0; i < vec.size(); i++)
	{
		vector<double> data = vec[i];

		cout << "\t=== Level: " << i <<  " ===" << endl;
		vdump(data);
	}
}

void init()
{

	if (values_all.size() > 0)
	{
		return;
	}

	values_all.resize(level_count);
	heights_all.resize(level_count);

	size_t factor = 0;

	for (size_t i = 0; i < level_count; i++)
	{
		vector<double> values, heights;

		if (i == 3)
		{
			factor = 2;
		}

		for (size_t j = 0; j < arr_size; j++)
		{
	   		values.push_back(static_cast<double> (6*i+5*j+51 - 3 * j * factor));
	  		heights.push_back(static_cast<double> ((1000+(j*3)) - i*75));
		}

		values_all[i] = values;
		heights_all[i] = heights;
	}

/*
	cout << "\t=== VALUES ===\n";
	dump(values_all);
	cout << "\t===HEIGHTS ===\n";
	dump(heights_all);
	cout << endl;
*/
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

	// vdump(result);
	BOOST_REQUIRE(result[1] == 56);
	BOOST_REQUIRE(result[19] == 50);
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

	// vdump(result);
	BOOST_REQUIRE(result[0] == 69);
	BOOST_REQUIRE(result[7] == 98);

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
	BOOST_REQUIRE(result[19] == 50);
	BOOST_REQUIRE(result[arr_size + 19] == 158);

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

	// vdump(result);
	BOOST_REQUIRE(result[2] == 268);
	BOOST_REQUIRE(result[18] == 492);

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

	// vdump(result);
	BOOST_REQUIRE(result[2] == 67);
	BOOST_REQUIRE(result[18] == 123);

}

BOOST_AUTO_TEST_CASE(MODIFIER_COUNT)
{

	init();

	auto mod = unique_ptr<modifier_count> (new modifier_count);

	vector<double> findv = {
		55, 15, 0, 0, 60,
		52, 3, 15, 10, 100,
		103, 0, 116, 50, 50,
		50, 10, 50, 1130, 111
	};

	// vdump(findv);

	mod->FindValue(findv);

	for (size_t i = 0; i < level_count; i++)
	{
		mod->Process(values_all[i], heights_all[i]);
	}

	auto values = mod->Result();

	// vdump(values);
	BOOST_REQUIRE(values[0] == 1);
	BOOST_REQUIRE(values[10] == 2);
	BOOST_REQUIRE(values[15] == 0);
	
}

BOOST_AUTO_TEST_CASE(MODIFIER_FINDHEIGHT)
{

	init();

	auto mod = unique_ptr<modifier_findheight> (new modifier_findheight);

	vector<double> findv = {
		55, 15, 0, 0, 60,
		52, 3, 15, 10, 100,
		103, 0, 116, 50, 50,
		50, 10, 50, 1130, 111
	};

	// vdump(findv);

	mod->FindValue(findv);

	for (size_t i = 0; i < level_count; i++)
	{
		mod->Process(values_all[i], heights_all[i]);
	}

	auto values = mod->Result();

	//vdump(values);
	BOOST_REQUIRE(values[0] == 950);
	BOOST_REQUIRE(values[2] == kFloatMissing);
	BOOST_CHECK_CLOSE(values[19], 874.361, kEpsilon);

	mod->FindNth(2);
	mod->Clear();

	for (size_t i = 0; i < level_count; i++)
	{
		mod->Process(values_all[i], heights_all[i]);
	}

	values = mod->Result();

	// vdump(values);

	BOOST_REQUIRE(values[0] == kFloatMissing);
	BOOST_CHECK_CLOSE(values[10], 866.111, kEpsilon);

	mod->FindNth(0);
	mod->Clear();

	for (size_t i = 0; i < level_count; i++)
	{
		mod->Process(values_all[i], heights_all[i]);
	}

	values = mod->Result();

	// vdump(values);

	BOOST_REQUIRE(values[0] == 950);
	BOOST_CHECK_CLOSE(values[10], 866.111, kEpsilon);
	BOOST_CHECK_CLOSE(values[19], 874.361, kEpsilon);

}

BOOST_AUTO_TEST_CASE(MODIFIER_FINDVALUE)
{

	init();

	auto mod = unique_ptr<modifier_findvalue> (new modifier_findvalue);

	vector<double> findh = {
		997, 0, 0, 0, 60,
		52, 3, 15, 900, 100,
		103, 0, 116, 50, 50,
		50, 10, 50, 1130, 111
	};

	//vdump(findh);

	mod->FindValue(findh);

	for (size_t i = 0; i < level_count; i++)
	{
		mod->Process(values_all[i], heights_all[i]);
	}

	auto values = mod->Result();

	//vdump(values);
	BOOST_REQUIRE(values[0] == 51.24);
	BOOST_REQUIRE(values[2] == kFloatMissing);
	BOOST_REQUIRE(values[8] == 100.92);

	mod->FindNth(0);
	mod->Clear();

	for (size_t i = 0; i < level_count; i++)
	{
		mod->Process(values_all[i], heights_all[i]);
	}
	
}