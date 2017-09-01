/**
 * @file weather_symbol.cpp
 *
 */

#include "weather_symbol.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include <map>

using namespace std;
using namespace himan::plugin;

const int rain[4][3] = {{1, 2, 3}, {0, 21, 31}, {0, 22, 32}, {0, 23, 33}};
const int thunder_and_rain[4][3] = {{1, 2, 3}, {0, 21, 31}, {0, 61, 61}, {0, 62, 62}};
const int snow[4][3] = {{1, 2, 3}, {0, 41, 51}, {0, 42, 52}, {0, 43, 53}};

weather_symbol::weather_symbol()
{
	itsLogger = logger("weather_symbol");

	// hilake: etsi_pilvi.F
	cloudMap.insert(std::make_pair(0, 1));
	cloudMap.insert(std::make_pair(704, 0));
	cloudMap.insert(std::make_pair(1301, 1));
	cloudMap.insert(std::make_pair(1302, 2));
	cloudMap.insert(std::make_pair(1303, 2));
	cloudMap.insert(std::make_pair(1305, 2));
	cloudMap.insert(std::make_pair(1309, 2));
	cloudMap.insert(std::make_pair(1403, 2));
	cloudMap.insert(std::make_pair(1501, 1));
	cloudMap.insert(std::make_pair(2307, 2));
	cloudMap.insert(std::make_pair(2403, 2));
	cloudMap.insert(std::make_pair(2305, 2));
	cloudMap.insert(std::make_pair(2501, 2));
	cloudMap.insert(std::make_pair(2302, 2));
	cloudMap.insert(std::make_pair(2303, 2));
	cloudMap.insert(std::make_pair(2304, 2));
	cloudMap.insert(std::make_pair(3309, 3));
	cloudMap.insert(std::make_pair(3307, 3));
	cloudMap.insert(std::make_pair(3604, 3));
	cloudMap.insert(std::make_pair(3405, 3));
	cloudMap.insert(std::make_pair(3306, 3));
	cloudMap.insert(std::make_pair(3502, 2));
}

void weather_symbol::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to weather_symbol
	 */

	SetParams({param("HESSAA-N", 338)});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void weather_symbol::Calculate(shared_ptr<info> myTargetInfo, unsigned short theThreadIndex)
{
	// Required source parameters

	param CParam("CLDSYM-N");
	param RTParam("HSADE1-N");

	level HLevel(himan::kHeight, 0, "HEIGHT");

	auto myThreadedLogger = logger("weather_symbolThread #" + to_string(theThreadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                      static_cast<string>(forecastLevel));

	info_t CInfo = Fetch(forecastTime, HLevel, CParam, forecastType, false);
	info_t RTInfo = Fetch(forecastTime, HLevel, RTParam, forecastType, false);

	if (!CInfo || !RTInfo)
	{
		itsLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " + static_cast<string>(forecastLevel));
		return;
	}

	string deviceType = "CPU";

	LOCKSTEP(myTargetInfo, CInfo, RTInfo)
	{
		double cloudSymbol = CInfo->Value();
		double rainType = RTInfo->Value();

		if (IsMissingValue({cloudSymbol, rainType}))
		{
			continue;
		}

		double weather_symbol;
		double ctype, rtype, rform, wtype;

		weather_symbol = 0;
		ctype = cloud_type(cloudSymbol);
		rtype = rain_type(rainType);
		rform = rain_form(rainType);
		wtype = weather_type(rainType);

		if (rform == 1)
		{
			if (rtype == 1)
			{
				if (ctype > 0 && wtype > 0)
				{
					// jatkuva vesisade
					weather_symbol = rain[static_cast<int>(wtype) - 1][static_cast<int>(ctype) - 1];
				}
			}
			else if (rtype == 2)
			{
				if (ctype > 0 && wtype > 0)
				{
					// kuurottainen vesisade
					weather_symbol = rain[static_cast<int>(wtype) - 1][static_cast<int>(ctype) - 1];
				}
			}
			else if (rtype == 3)
			{
				if (ctype > 0 && wtype > 0)
				{
					// ukkonen ja vesisade
					weather_symbol = thunder_and_rain[static_cast<int>(wtype) - 1][static_cast<int>(ctype) - 1];
				}
			}
		}
		else if (rform == 2)
		{
			if (rtype == 1)
			{
				if (ctype > 0 && wtype > 0)
				{
					// jatkuva lumisade
					weather_symbol = snow[static_cast<int>(wtype) - 1][static_cast<int>(ctype) - 1];
				}
			}
			else if (rtype == 2)
			{
				if (ctype > 0 && wtype > 0)
				{
					// kuurottainen lumisade
					weather_symbol = snow[static_cast<int>(wtype) - 1][static_cast<int>(ctype) - 1];
				}
			}
			else if (rtype == 3)
			{
				if (ctype > 0 && wtype > 0)
				{
					// ukkonen ja lumisade
					weather_symbol = snow[static_cast<int>(wtype) - 1][static_cast<int>(ctype) - 1];
				}
			}
		}

		myTargetInfo->Value(weather_symbol);
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}

double weather_symbol::rain_form(double rr)
{
	double rain_form;
	if (rr <= 67 || rr == 80 || rr == 81 || rr == 82 || rr == 95 || rr == 97)
	{
		rain_form = 1;
	}
	else
	{
		rain_form = 2;
	}
	return rain_form;
}

double weather_symbol::weather_type(double rr)
{
	double weather_type;
	// or maybe std::map?
	if (rr >= 50 && rr <= 54)
	{
		weather_type = 2;
	}
	else if (rr >= 56 && rr <= 61)
	{
		weather_type = 2;
	}
	else if (rr >= 76 && rr <= 80)
	{
		weather_type = 2;
	}
	else if (rr == 66 || rr == 70 || rr == 71 || rr == 85 || rr == 2070)
	{
		weather_type = 2;
	}
	else if (rr == 55 || rr == 62 || rr == 63 || rr == 67 || rr == 68 || rr == 72 || rr == 73 || rr == 81 || rr == 95)
	{
		weather_type = 3;
	}
	else if (rr == 64 || rr == 65 || rr == 69 || rr == 74 || rr == 75 || rr == 82 || rr == 86 || rr == 97)
	{
		weather_type = 4;
	}
	else if (rr == 0)
	{
		weather_type = 1;
	}
	else
	{
		throw runtime_error("Invalid value for rr: Fix me!");
	}

	return weather_type;
}

double weather_symbol::rain_type(double rr)
{
	double rain_type;
	if (rr <= 79 || rr == 2070)
	{
		rain_type = 1;
	}
	else if (rr >= 80 && rr <= 86)
	{
		rain_type = 2;
	}
	else if (rr == 95 || rr == 97)
	{
		rain_type = 3;
	}
	else
	{
		throw runtime_error("Invalid value for rr: Fix me!");
	}

	return rain_type;
}

double weather_symbol::cloud_type(double cloud)
{
	// etsi_pilvityyppi
	typedef std::map<double, double>::const_iterator MapIter;
	for (MapIter iter = cloudMap.begin(); iter != cloudMap.end(); ++iter)
	{
		if (iter->first == cloud)
		{
			return iter->second;
		}
	}
	return 0;
}
