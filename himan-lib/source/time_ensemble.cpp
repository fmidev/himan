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

	if (itsMaximumMissingForecasts > 0)
	{
		if (numMissingForecasts > itsMaximumMissingForecasts)
		{
			itsLogger->Fatal("Maximum number of missing fields (" + std::to_string(itsMaximumMissingForecasts) +
			                 ") reached, aborting");
			exit(1);
		}
	}
	else
	{
		if (numMissingForecasts > 0)
		{
			itsLogger->Fatal("Missing " + std::to_string(numMissingForecasts) + " of " +
			                 std::to_string(itsMaximumMissingForecasts) + " allowed missing fields of data");
			exit(1);
		}
	}

	itsLogger->Info("Succesfully loaded " + std::to_string(itsForecasts.size()) + "/" +
	                std::to_string(itsExpectedEnsembleSize) + " fields");

}
