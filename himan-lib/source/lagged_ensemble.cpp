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
lagged_ensemble::lagged_ensemble(const param& parameter, size_t expectedEnsembleSize, HPTimeResolution lagResolution,
                                 int lag, size_t numberOfSteps)
    : itsLagResolution(lagResolution), itsLag(lag), itsNumberOfSteps(numberOfSteps)
{
	itsParam = parameter;
	itsExpectedEnsembleSize = expectedEnsembleSize;
	itsEnsembleType = kLaggedEnsemble;

	itsDesiredForecasts.reserve(expectedEnsembleSize);
	itsDesiredForecasts.push_back(forecast_type(kEpsControl, 0));

	for (size_t i = 1; i < itsDesiredForecasts.capacity(); i++)
	{
		itsDesiredForecasts.push_back(forecast_type(kEpsPerturbation, static_cast<double>(i)));
	}

	itsLogger = logger("lagged_ensemble");

	itsForecasts.reserve(itsExpectedEnsembleSize * itsNumberOfSteps);
}

// We do a 'full fetch' every time, relying on himan's cache to store the previously stored
// infos. This makes the code some what simpler, since we don't have to deal with data
// stored in different timesteps.
void lagged_ensemble::Fetch(std::shared_ptr<const plugin_configuration> config, const forecast_time& time,
                            const level& forecastLevel)
{
	assert(itsLag < 0);
	assert(itsNumberOfSteps > 0);

	auto f = GET_PLUGIN(fetcher);

	itsForecasts.clear();

	const int lag = itsLag;
	int missing = 0;
	int loaded = 0;

	itsLogger.Info("Fetching for " + std::to_string(itsNumberOfSteps) + " timesteps with lag " + std::to_string(lag));

	// Start from the 'earliest' origin time
	for (int currentStep = static_cast<int>(itsNumberOfSteps) - 1; currentStep >= 0; currentStep--)
	{
		forecast_time ftime(time);

		if (currentStep != 0) ftime.OriginDateTime().Adjust(itsLagResolution, lag * currentStep);

		// Missing forecasts are only checked for the current origin time, not for lagged
		for (const auto& desired : itsDesiredForecasts)
		{
			try
			{
				auto Info = f->Fetch(config, ftime, forecastLevel, itsParam, desired, false);
				itsForecasts.push_back(Info);

				if (currentStep == 0) loaded++;
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
					if (currentStep == 0) missing++;
				}
			}
		}
	}
	VerifyValidForecastCount(loaded, missing);
}

void lagged_ensemble::VerifyValidForecastCount(int numLoadedForecasts, int numMissingForecasts)
{
	if (itsMaximumMissingForecasts > 0)
	{
		if (numMissingForecasts >= itsMaximumMissingForecasts)
		{
			itsLogger.Fatal("maximum number of missing fields " + std::to_string(numMissingForecasts) + "/" +
			                std::to_string(itsMaximumMissingForecasts) + " reached, aborting");
			abort();
		}
	}
	else
	{
		if (numMissingForecasts > 0)
		{
			itsLogger.Fatal("missing " + std::to_string(numMissingForecasts) + " of " +
			                std::to_string(itsMaximumMissingForecasts) + " allowed missing fields of data");
			abort();
		}
	}
	itsLogger.Info("succesfully loaded " + std::to_string(numLoadedForecasts) + "/" +
	               std::to_string(itsDesiredForecasts.size()) + " fields");
}

HPTimeResolution lagged_ensemble::LagResolution() const { return itsLagResolution; }
int lagged_ensemble::Lag() const { return itsLag; }
size_t lagged_ensemble::NumberOfSteps() const { return itsNumberOfSteps; }
}  // namespace himan
