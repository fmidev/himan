#include "himan_unit.h"
#include "plugin_factory.h"
#include "latitude_longitude_grid.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "querydata.h"

#undef HIMAN_AUXILIARY_INCLUDE

#define BOOST_TEST_MODULE querydata

using namespace std;
using namespace himan;

BOOST_AUTO_TEST_CASE(QUERYDATA)
{
	// Create himan::info, convert it to querydata, convert it back to info and compare results

	// Create info

	auto newGrid = make_shared<latitude_longitude_grid> ();
	info newInfo;

	newInfo.Producer(producer(230, 86, 203, "TEST"));

	vector<forecast_time> times;
	vector<level> levels;
	vector<param> params;
	vector<forecast_type> types;

	times.push_back(forecast_time("2014-04-04 00:00:00", "2014-04-04 01:00:00"));
	levels.push_back(level(kHeight, 2, "Height"));
	params.push_back(param("TestParam", 12));
	types.push_back(forecast_type(kDeterministic));

	newInfo.Times(times);
	newInfo.Levels(levels);
	newInfo.Params(params);
	newInfo.ForecastTypes(types);

	newGrid->ScanningMode(himan::kBottomLeft);
	newGrid->BottomLeft(point(20,50));
	newGrid->TopRight(point(25,70));

	const size_t ni = 10, nj = 20;

	newGrid->Ni(ni);
	newGrid->Nj(nj);

	auto data = matrix<double>(ni, nj, 1, himan::kFloatMissing);

	for (size_t i = 0; i < ni*nj; i++)
	{
		data.Set(i, static_cast<double> (i+1));
	}

	newGrid->Data(data);
	newInfo.Create(newGrid.get());

	// Convert info to querydata

	auto q = dynamic_pointer_cast <plugin::querydata> (plugin_factory::Instance()->Plugin("querydata")); 

	auto qdata = q->CreateQueryData(newInfo, false);

	BOOST_REQUIRE(qdata);

	// Convect querydata back to info

	auto convertInfo = q->CreateInfo(qdata);

	BOOST_REQUIRE(*newInfo.Grid() == *convertInfo->Grid());
//	BOOST_REQUIRE(newInfo->Data() == convertInfo->Data());

}
