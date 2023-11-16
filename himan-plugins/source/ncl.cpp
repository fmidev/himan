/**
 * @file ncl.cpp
 *
 */

#include "ncl.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "plugin_factory.h"

#include "hitool.h"

using namespace std;
using namespace himan::plugin;

const string itsName("ncl");

ncl::ncl() : itsTargetTemperature(kHPMissingInt)
{
	itsLogger = logger(itsName);
}

void ncl::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	param theRequestedParam;

	if (itsConfiguration->Exists("temp") && itsConfiguration->GetValue("temp") == "-20")
	{
		theRequestedParam.Name("HM20C-M");
		itsTargetTemperature = -20;
	}

	if (itsConfiguration->Exists("temp") && itsConfiguration->GetValue("temp") == "0")
	{
		theRequestedParam.Name("H0C-M");
		itsTargetTemperature = 0;
	}

	if (theRequestedParam.Name() == "XX-X")
	{
		itsLogger.Fatal("Requested temperature not set");
		exit(1);
	}

	SetParams({theRequestedParam});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void ncl::Calculate(shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex)
{
	const param TParam("T-K");

	auto myThreadedLogger = logger(itsName + "Thread #" + to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                      static_cast<string>(forecastLevel));

	auto h = GET_PLUGIN(hitool);

	h->Configuration(itsConfiguration);
	h->Time(forecastTime);
	h->ForecastType(forecastType);

	auto result = h->VerticalHeight<double>(TParam, 0, 12000, itsTargetTemperature + constants::kKelvin, 1);
	myTargetInfo->Data().Set(result);

	myThreadedLogger.Info("[CPU] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) + "/" +
	                      to_string(myTargetInfo->Data().Size()));
}
