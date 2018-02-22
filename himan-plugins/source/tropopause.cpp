#include "tropopause.h"
#include "fetcher.h"
#include "forecast_time.h"
#include "hitool.h"
#include "info.h"
#include "logger.h"
#include "metutil.h"
#include "plugin_factory.h"
#include <algorithm>

using namespace std;
using namespace himan;
using namespace himan::plugin;

tropopause::tropopause()
{
	itsLogger = logger("tropopause");
}
void tropopause::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	param TR("TR-FL");

	SetParams({TR});

	Start();
}

void tropopause::Calculate(info_t myTargetInfo, unsigned short threadIndex)
{
	const param H("HL-M");
	const param T("T-K");

	auto h = GET_PLUGIN(hitool);
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());
	h->HeightUnit(kHPa);

	//Search is limited to the interval FL140-FL530 i.e. 600-100 HPa
	auto FL530 = h->LevelForHeight(myTargetInfo->Producer(), 100.);
	auto FL140 = h->LevelForHeight(myTargetInfo->Producer(), 600.);

	auto myThreadedLogger = logger("tropopause_pluginThread #" + to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Debug("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()));

	size_t firstLevel = static_cast<size_t>(FL140.second.Value());
	size_t lastLevel = static_cast<size_t>(FL530.first.Value());

	// fetch all data first to vectors representing vertical dimension
	vector<vector<double>> height;
	vector<vector<double>> temp;
        vector<vector<double>> pres;

	for (size_t lvl = firstLevel; lvl > lastLevel; --lvl)
	{
		height.push_back(
		    VEC(Fetch(forecastTime, level(kHybrid, static_cast<double>(lvl)), param("HL-M"), forecastType, false)));
		temp.push_back(
		    VEC(Fetch(forecastTime, level(kHybrid, static_cast<double>(lvl)), param("T-K"), forecastType, false)));
		pres.push_back(
                    VEC(Fetch(forecastTime, level(kHybrid, static_cast<double>(lvl)), param("P-HPA"), forecastType, false)));

	}

	size_t grd_size = temp[0].size();
	vector<double> tropopause(grd_size, MissingDouble());

	// outer loop goes horizontal direction
	for (size_t i = 0; i < grd_size; ++i)
	{
		// inner loop goes vertical and searches for lapse rate smaller 2K/km and check average lapse rate to all levels within 2km above is also smaller 2K/km
		for (size_t j = 1; j < firstLevel - lastLevel - 1; ++j)
		{
			const double lapseRate = -1000.0*(temp[j + 1][i] - temp[j - 1][i]) / (height[j + 1][i] - height[j - 1][i]);
			if (lapseRate <= 2.0 )
			{
				// set tropopause height
				tropopause[i] = 100.*pres[j][i];

				// check 2km above condition
				size_t k = j + 1;
				while (height[k][i] - height[j][i] <= 2000.0 && k < firstLevel - lastLevel-1)
				{
					if (-1000.0*(temp[k][i] - temp[j][i]) / (height[k][i] - height[j][i]) > 2.0)
					{
						tropopause[i] = MissingDouble();
						break;
					}
					++k;
				}
			}
			if(IsValid(tropopause[i])) break;
		}
	}

	// convert pressure to flight level
	transform(tropopause.begin(), tropopause.end(), tropopause.begin(), metutil::FlightLevel_);

	myTargetInfo->ParamIndex(0);
	myTargetInfo->Grid()->Data().Set(move(tropopause));

	string deviceType = "CPU";
	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
