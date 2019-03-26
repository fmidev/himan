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

	ASSERT(ftime.Step() >= 0);
}

std::vector<forecast_time> CreateTimeList(const forecast_time& origtime, size_t primaryTimeMaskLen,
                                          HPTimeResolution primaryTimeSpan, int secondaryTimeMaskLen,
                                          HPTimeResolution secondaryTimeSpan, int secondaryTimeMaskStep,
                                          bool isCumulative)
{
	std::vector<forecast_time> ret;

	auto ftime = origtime;

	for (size_t i = 0; i < primaryTimeMaskLen; i++)
	{
		for (int j = secondaryTimeMaskLen; j >= -secondaryTimeMaskLen; j -= secondaryTimeMaskStep)
		{
			auto curtime = ftime;

			curtime.ValidDateTime().Adjust(secondaryTimeSpan, j);

			if (isCumulative)
			{
				ASSERT(secondaryTimeSpan == kHourResolution);
				while (curtime.Step() < 1)
				{
					curtime.OriginDateTime().Adjust(secondaryTimeSpan, -12);
				}

				while (curtime.Step() > 12)
				{
					curtime.OriginDateTime().Adjust(secondaryTimeSpan, 12);
				}
			}
			else
			{
				curtime.OriginDateTime().Adjust(secondaryTimeSpan, j);
			}

			ret.push_back(curtime);
		}

		AdjustTimes(ftime, primaryTimeSpan, -1);
	}

	return ret;
}

time_ensemble::time_ensemble(const param& parameter) : itsPrimaryTimeSpan(kYearResolution)
{
	itsParam = parameter;
	itsExpectedEnsembleSize = 0;
	itsEnsembleType = kTimeEnsemble;

	itsLogger = logger("time_ensemble");
}

time_ensemble::time_ensemble(const param& parameter, size_t primaryTimeMaskLen, HPTimeResolution primaryTimeSpan,
                             int secondaryTimeMaskLen, int secondaryTimeMaskStep, HPTimeResolution secondaryTimeSpan)
    : itsPrimaryTimeSpan(primaryTimeSpan),
      itsSecondaryTimeMaskLen(secondaryTimeMaskLen),
      itsSecondaryTimeMaskStep(secondaryTimeMaskStep),
      itsSecondaryTimeSpan(secondaryTimeSpan)
{
	itsParam = parameter;
	itsExpectedEnsembleSize = primaryTimeMaskLen;
	itsEnsembleType = kTimeEnsemble;

	// itsDesiredForecasts is not used in time_ensemble directly,
	// but ensemble uses it at least in VerifyValidForecastCount()
	itsDesiredForecasts.resize(itsExpectedEnsembleSize *
	                           (2 * (itsSecondaryTimeMaskLen / itsSecondaryTimeMaskStep) + 1));

	itsLogger = logger("time_ensemble");
}

void time_ensemble::Fetch(std::shared_ptr<const plugin_configuration> config, const forecast_time& time,
                          const level& forecastLevel)
{
	auto f = GET_PLUGIN(fetcher);

	forecast_time ftime(time);

	itsForecasts.clear();
	int numMissingForecasts = 0;

	auto timeList = CreateTimeList(ftime, itsExpectedEnsembleSize, itsPrimaryTimeSpan, itsSecondaryTimeMaskLen,
	                               itsSecondaryTimeSpan, itsSecondaryTimeMaskStep,
	                               (itsParam.Name() == "FFG-MS" || itsParam.Name() == "RRR-KGM2"));

	// randomize timelist so that different threads start to fetch different data
	std::random_shuffle(timeList.begin(), timeList.end());

	for (const auto& tm : timeList)
	{
		try
		{
			auto info = f->Fetch<float>(config, tm, forecastLevel, itsParam);

			itsForecasts.push_back(info);
		}
		catch (HPExceptionType& e)
		{
			if (e != kFileDataNotFound)
			{
				itsLogger.Fatal("Unable to proceed");
				himan::Abort();
			}
			else
			{
				numMissingForecasts++;
			}
		}
	}

	VerifyValidForecastCount(numMissingForecasts);
}
