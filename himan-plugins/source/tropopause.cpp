#include "tropopause.h"
#include "fetcher.h"
#include "forecast_time.h"
#include "hitool.h"
#include "info.h"
#include "logger.h"
#include "metutil.h"
#include "plugin_factory.h"
#include "radon.h"
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

	param TR("TROPO-FL");

	SetParams({TR});

	Start();
}

void tropopause::Calculate(shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex)
{
	const param H("HL-M");
	const param T("T-K");
	const param P("P-HPA");

	auto h = GET_PLUGIN(hitool);
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());
	h->HeightUnit(kHPa);

	// Search is limited to the interval FL140-FL530 i.e. 600-100 HPa
	auto FL530 = h->LevelForHeight(myTargetInfo->Producer(), 100., itsConfiguration->TargetGeomName());
	auto FL140 = h->LevelForHeight(myTargetInfo->Producer(), 600., itsConfiguration->TargetGeomName());

	auto myThreadedLogger = logger("tropopause_pluginThread #" + to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Debug("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()));

	size_t firstLevel = static_cast<size_t>(FL140.first.Value());
	size_t lastLevel = static_cast<size_t>(FL530.second.Value());

	size_t lvl_size = firstLevel - lastLevel;

	// fetch all data first to vectors representing vertical dimension
	vector<vector<double>> height;
	vector<vector<double>> temp;
	vector<vector<double>> pres;

	auto r = GET_PLUGIN(radon);

	const std::string lt = r->RadonDB().GetProducerMetaData(myTargetInfo->Producer().Id(), "hybrid level type");

	const HPLevelType hybridLevelType = (lt.empty()) ? kHybrid : HPStringToLevelType.at(lt);

	level curLevel;
	double firstValue = static_cast<double>(firstLevel);

	switch (hybridLevelType)
	{
		case kHybrid:
		default:
			curLevel = level(hybridLevelType, firstValue);
			break;
		case kGeneralizedVerticalLayer:
			curLevel = level(hybridLevelType, firstValue, firstValue + 1);
			break;
	}

	while (curLevel.Value() > static_cast<double>(lastLevel))
	{
		auto heightInfo = Fetch(forecastTime, curLevel, H, forecastType, false);
		auto tempInfo = Fetch(forecastTime, curLevel, T, forecastType, false);
		auto presInfo = Fetch(forecastTime, curLevel, P, forecastType, false);

		level::EqualAdjustment(curLevel, -1);

		if (!(heightInfo && tempInfo && presInfo))
		{
			--lvl_size;
			continue;
		}

		height.push_back(VEC(heightInfo));
		temp.push_back(VEC(tempInfo));
		pres.push_back(VEC(presInfo));
	}

	if (height.empty())
	{
		return;
	}

	size_t grd_size = myTargetInfo->SizeLocations();
	vector<double> result(grd_size, MissingDouble());

	// outer loop goes horizontal direction
	for (size_t i = 0; i < grd_size; ++i)
	{
		// inner loop goes vertical and searches for lapse rate smaller 2K/km and check average lapse rate to all levels
		// within 2km above is also smaller 2K/km
		for (size_t j = 0; j < lvl_size - 1; ++j)
		{
			// Search is limited to the interval FL140-FL530 i.e. 600-100 HPa
			if (pres[j][i] < 100. || pres[j][i] > 600.)
				continue;

			const double lapseRate = -1000.0 * (temp[j + 1][i] - temp[j][i]) / (height[j + 1][i] - height[j][i]);
			if (lapseRate <= 2.0)
			{
				// set tropopause height
				result[i] = 100. * pres[j][i];

				// check 2km above condition
				size_t k = j + 1;
				while (height[k][i] - height[j][i] <= 2000.0 && k < lvl_size - 1)
				{
					if (-1000.0 * (temp[k][i] - temp[j][i]) / (height[k][i] - height[j][i]) > 2.0)
					{
						result[i] = MissingDouble();
						break;
					}
					++k;
				}
			}
			if (IsValid(result[i]))
				break;
		}
	}

	// convert pressure to flight level
	transform(result.begin(), result.end(), result.begin(), metutil::FlightLevel_);
	myTargetInfo->Data().Set(move(result));

	string deviceType = "CPU";
	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
