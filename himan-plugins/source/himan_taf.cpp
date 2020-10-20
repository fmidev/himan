#include "himan_taf.h"
#include "fetcher.h"
#include "forecast_time.h"
#include "hitool.h"
#include "info.h"
#include "level.h"
#include "logger.h"
#include "plugin_factory.h"
#include "radon.h"
#include "util.h"
#include <algorithm>

using namespace std;
using namespace himan;
using namespace himan::plugin;

himan_taf::himan_taf() : itsStrictMode(false)
{
	itsLogger = logger("himan_taf");
}
void himan_taf::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	param CL1("CL1-FT");
	param CL2("CL2-FT");
	param CL3("CL3-FT");
	param CL4("CL4-FT");

	param CC1("CL1-PRCNT");
	param CC2("CL2-PRCNT");
	param CC3("CL3-PRCNT");
	param CC4("CL4-PRCNT");

	SetParams({CL1, CC1, CL2, CC2, CL3, CC3, CL4, CC4});

	Start();
}

void himan_taf::Calculate(info_t myTargetInfo, unsigned short threadIndex)
{
	auto myThreadedLogger = logger("himan_taf_pluginThread #" + to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();

	myThreadedLogger.Debug("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()));

	// find height of cb base and convert to feet

	auto h = GET_PLUGIN(hitool);
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());
	h->HeightUnit(kM);

	// upper cloud search
	double max_h = 12000.0;

	// search for lowest cloud layer
	auto cl1b = h->VerticalHeight<double>(param("N-0TO1"), 0.0, max_h, 0.05, 1);
	auto cl1t = h->VerticalHeight<double>(param("N-0TO1"), 0.0, max_h, 0.05, 2);

	// second lowest
	auto cl2b = h->VerticalHeight<double>(param("N-0TO1"), 0.0, max_h, 0.05, 3);
	auto cl2t = h->VerticalHeight<double>(param("N-0TO1"), 0.0, max_h, 0.05, 4);

	// third lowest
	auto cl3b = h->VerticalHeight<double>(param("N-0TO1"), 0.0, max_h, 0.05, 5);
	auto cl3t = h->VerticalHeight<double>(param("N-0TO1"), 0.0, max_h, 0.05, 6);

	// fourth lowest
	auto cl4b = h->VerticalHeight<double>(param("N-0TO1"), 0.0, max_h, 0.05, 7);
	auto cl4t = h->VerticalHeight<double>(param("N-0TO1"), 0.0, max_h, 0.05, 8);

	auto N1 = h->VerticalMaximum<double>(param("N-0TO1"), cl1b, cl1t);
	auto N2 = h->VerticalMaximum<double>(param("N-0TO1"), cl2b, cl2t);
	auto N3 = h->VerticalMaximum<double>(param("N-0TO1"), cl3b, cl3t);
	auto N4 = h->VerticalMaximum<double>(param("N-0TO1"), cl4b, cl4t);

	// end find height of cb base

	myTargetInfo->Index<param>(0);
	myTargetInfo->Grid()->Data().Set(move(cl1b));

	myTargetInfo->Index<param>(1);
	myTargetInfo->Grid()->Data().Set(move(N1));

	myTargetInfo->Index<param>(2);
	myTargetInfo->Grid()->Data().Set(move(cl2b));

	myTargetInfo->Index<param>(3);
	myTargetInfo->Grid()->Data().Set(move(N2));

	myTargetInfo->Index<param>(4);
	myTargetInfo->Grid()->Data().Set(move(cl3b));

	myTargetInfo->Index<param>(5);
	myTargetInfo->Grid()->Data().Set(move(N3));

	myTargetInfo->Index<param>(6);
	myTargetInfo->Grid()->Data().Set(move(cl4b));

	myTargetInfo->Index<param>(7);
	myTargetInfo->Grid()->Data().Set(move(N4));

	string deviceType = "CPU";
	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
