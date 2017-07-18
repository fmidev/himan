#include "time_ensemble.h"

#include "plugin_factory.h"

#define HIMAN_AUXILIARY_INCLUDE
#include "fetcher.h"
#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan;
using namespace himan::plugin;

void AdjustTimes(forecast_time& ftime, HPTimeResolution timeSpan, int value)
{
	ftime.OriginDateTime().Adjust(timeSpan, value);
	ftime.ValidDateTime().Adjust(timeSpan, value);

	if (ftime.OriginDateTime().IsLeapYear() && timeSpan == kYearResolution)
	{
		// case:
		// origin time: 2017-02-28 12
		// valid time: 2017-03-01 00 (step: 12hrs)
		//
		// after adjustment of -1 years:
		// origin time: 2016-02-28 12
		// valid time: 2016-03-01 00 (step: 36hrs)
		//
		// fix by adjusting valid time -1 day
		// valid time: 2016-02-29 00 (step: 12hrs)

		ftime.ValidDateTime().Adjust(kDayResolution, -1);
	}

	if (ftime.Step() < 0)
	{
		// to recover from a leap year extra day subtraction,
		// to continue from previous example:
		// origin time: 2016-02-28 12
		// valid time: 2016-02-29 00 (step: 12hrs)
		//
		// after adjustment of -1 years:
		// origin time: 2015-02-28 12
		// valid time: 2015-02-28 00 (step: -12hrs)
		//
		// fix by adjusting valid time +1 day

		ftime.ValidDateTime().Adjust(kDayResolution, 1);
	}

	assert(ftime.Step() >= 0);
}

time_ensemble::time_ensemble(const param& parameter) : itsTimeSpan(kYearResolution)
{
	itsParam = parameter;
	itsExpectedEnsembleSize = 0;
	itsEnsembleType = kTimeEnsemble;

	itsLogger = logger("time_ensemble");
}

time_ensemble::time_ensemble(const param& parameter, size_t expectedEnsembleSize, HPTimeResolution theTimeSpan)
    : itsTimeSpan(theTimeSpan)
{
	itsParam = parameter;
	itsExpectedEnsembleSize = expectedEnsembleSize;
	itsEnsembleType = kTimeEnsemble;

	// itsDesiredForecasts is not used in time_ensemble directly,
	// but ensemble uses it at least in VerifyValidForecastCount()
	itsDesiredForecasts.resize(itsExpectedEnsembleSize);

	itsLogger = logger("time_ensemble");
}

void time_ensemble::Fetch(std::shared_ptr<const plugin_configuration> config, const forecast_time& time,
                          const level& forecastLevel)
{
	auto f = GET_PLUGIN(fetcher);

	forecast_time ftime(time);

	itsForecasts.clear();
	int numMissingForecasts = 0;

	for (size_t i = 0; i < itsExpectedEnsembleSize; i++)
	{
		try
		{
			auto info = f->Fetch(config, ftime, forecastLevel, itsParam);

			AdjustTimes(ftime, itsTimeSpan, -1);

			itsForecasts.push_back(info);
		}
		catch (HPExceptionType& e)
		{
			if (e != kFileDataNotFound)
			{
				itsLogger.Fatal("Unable to proceed");
				abort();
			}
			else
			{
				numMissingForecasts++;

				AdjustTimes(ftime, itsTimeSpan, -1);
			}
		}
	}

	VerifyValidForecastCount(numMissingForecasts);
}
