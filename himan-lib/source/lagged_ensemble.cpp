#include "lagged_ensemble.h"

#include "plugin_factory.h"
#include "util.h"

#define HIMAN_AUXILIARY_INCLUDE
#include "fetcher.h"
#undef HIMAN_AUXILIARY_INCLUDE

#include <math.h>

using namespace himan;
using namespace himan::plugin;
using namespace himan::util;

namespace himan
{
lagged_ensemble::lagged_ensemble(const param& parameter, size_t expectedEnsembleSize, const time_duration& theLag,
                                 size_t numberOfSteps)
    : lagged_ensemble(parameter, expectedEnsembleSize, theLag * static_cast<int>(numberOfSteps - 1), theLag * -1)
{
}

lagged_ensemble::lagged_ensemble(const param& parameter, size_t expectedEnsembleSize, const time_duration& theLag,
                                 const time_duration& theStep)
    : itsLag(theLag), itsStep(theStep)
{
	itsParam = parameter;
	itsEnsembleType = kLaggedEnsemble;

	itsDesiredForecasts.reserve(expectedEnsembleSize);
	itsDesiredForecasts.push_back(forecast_type(kEpsControl, 0));

	for (size_t i = 1; i < itsDesiredForecasts.capacity(); i++)
	{
		itsDesiredForecasts.push_back(forecast_type(kEpsPerturbation, static_cast<float>(i)));
	}

	itsLogger = logger("lagged_ensemble");

	if (itsLag.Hours() > 0)
	{
		itsLogger.Fatal("Lag has to be negative");
		himan::Abort();
	}
	if (itsStep.Hours() < 0)
	{
		itsLogger.Fatal("Step has to be positive (" + std::to_string(itsStep.Hours()) + ")");
		himan::Abort();
	}

	itsForecasts.reserve(2 * expectedEnsembleSize);  // educated guess
}

// We do a 'full fetch' every time, relying on himan's cache to store the previously stored
// infos. This makes the code some what simpler, since we don't have to deal with data
// stored in different timesteps.
void lagged_ensemble::Fetch(std::shared_ptr<const plugin_configuration> config, const forecast_time& time,
                            const level& forecastLevel)
{
	auto f = GET_PLUGIN(fetcher);

	itsForecasts.clear();

	int missing = 0;
	int loaded = 0;

	itsLogger.Info("Fetching with lag " + static_cast<std::string>(itsLag));

	forecast_time ftime(time);
	ftime.OriginDateTime() += itsLag;

	while (ftime.OriginDateTime() <= time.OriginDateTime())
	{
		// Missing forecasts are checked for both the current origin time, and for lagged
		for (const auto& desired : itsDesiredForecasts)
		{
			try
			{
				auto Info = f->Fetch<float>(config, ftime, forecastLevel, itsParam, desired, false);
				itsForecasts.push_back(Info);

				loaded++;
			}
			catch (HPExceptionType& e)
			{
				if (e != kFileDataNotFound)
				{
					itsLogger.Fatal("Unable to proceed");
					himan::Abort();
				}
				missing++;
			}
		}
		ftime.OriginDateTime() += itsStep;
	}

	VerifyValidForecastCount(loaded, missing);
}

void lagged_ensemble::VerifyValidForecastCount(int numLoadedForecasts, int numMissingForecasts)
{
	if (itsMaximumMissingForecasts > 0)
	{
		if (numMissingForecasts > itsMaximumMissingForecasts)
		{
			itsLogger.Fatal("Maximum number of missing fields " + std::to_string(numMissingForecasts) + "/" +
			                std::to_string(itsMaximumMissingForecasts) + " reached, aborting");
			throw kFileDataNotFound;
		}
	}
	else
	{
		if (numMissingForecasts > 0)
		{
			itsLogger.Fatal("Missing " + std::to_string(numMissingForecasts) + " of " +
			                std::to_string(itsMaximumMissingForecasts) + " allowed missing fields of data");
			throw kFileDataNotFound;
		}
	}
	itsLogger.Info("Succesfully loaded " + std::to_string(numLoadedForecasts) + " fields");
}

time_duration lagged_ensemble::Lag() const
{
	return itsLag;
}
}  // namespace himan
