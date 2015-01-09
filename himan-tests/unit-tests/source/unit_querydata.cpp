#include "himan_unit.h"
#include "plugin_factory.h"
#include "regular_grid.h"

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

	auto newGrid = make_shared<regular_grid> ();
	auto newInfo = make_shared<info> ();

	newInfo->Producer(producer(230, 86, 203, "TEST"));

	vector<forecast_time> times;
	vector<level> levels;
	vector<param> params;

	times.push_back(forecast_time("2014-04-04 00:00:00", "2014-04-04 01:00:00"));
	levels.push_back(level(kHeight, 2, "Height"));
	params.push_back(param("TestParam", 12));

	newInfo->Times(times);
	newInfo->Levels(levels);
	newInfo->Params(params);

	newGrid->ScanningMode(himan::kBottomLeft);
	newGrid->UVRelativeToGrid(false);

	newGrid->Projection(kLatLonProjection);
	newGrid->BottomLeft(point(20,50));
	newGrid->TopRight(point(25,70));

	size_t nx = 10, ny = 20;

	auto data = matrix<double>(nx, ny, 1, kFloatMissing);

	for (size_t i = 0; i < nx*ny; i++)
	{
		data.Set(i, static_cast<double> (i+1));
	}

	newGrid->Data(data);
	newInfo->Create(newGrid.get());

	// Convert info to querydata

	auto q = dynamic_pointer_cast <plugin::querydata> (plugin_factory::Instance()->Plugin("querydata")); 

	auto qdata = q->CreateQueryData(*newInfo, false);

	BOOST_REQUIRE(qdata);

	// Convect querydata back to info

	auto convertInfo = q->CreateInfo(qdata);

	BOOST_REQUIRE(*newInfo->Grid() == *convertInfo->Grid());
//	BOOST_REQUIRE(newInfo->Data() == convertInfo->Data());

}
