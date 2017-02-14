#include "time_ensemble.h"

#include "logger_factory.h"
#include "plugin_factory.h"

#define HIMAN_AUXILIARY_INCLUDE
#include "fetcher.h"
#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan;
using namespace himan::plugin;

time_ensemble::time_ensemble(const param& parameter) : itsTimeSpan(kYearResolution)
{
	itsParam = parameter;
	itsExpectedEnsembleSize = 0;
	itsEnsembleType = kTimeEnsemble;

	itsLogger = std::unique_ptr<logger>(logger_factory::Instance()->GetLog("time_ensemble"));
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

	itsLogger = std::unique_ptr<logger>(logger_factory::Instance()->GetLog("time_ensemble"));
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

			itsForecasts.push_back(info);

			ftime.OriginDateTime().Adjust(itsTimeSpan, -1);
			ftime.ValidDateTime().Adjust(itsTimeSpan, -1);
		}
		catch (HPExceptionType& e)
		{
			if (e != kFileDataNotFound)
			{
				itsLogger->Fatal("Unable to proceed");
				exit(1);
			}
			else
			{
				numMissingForecasts++;
				ftime.OriginDateTime().Adjust(itsTimeSpan, -1);
				ftime.ValidDateTime().Adjust(itsTimeSpan, -1);

			}
		}
	}

	VerifyValidForecastCount(numMissingForecasts);
}
