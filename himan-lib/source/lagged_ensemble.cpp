#include "lagged_ensemble.h"

#include "logger_factory.h"
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

// NOTE NOTE NOTE if we construct the object again always, we can't store any info about last fetch/previous fetch
// this means we can't use std::unique_ptr in a loop and we can't std::move it to some helper function!
lagged_ensemble::lagged_ensemble(const param& parameter, size_t expectedEnsembleSize, HPTimeResolution lagResolution,
                                 int lag, size_t numberOfSteps)
    : itsLagResolution(lagResolution), itsLag(lag), itsNumberOfSteps(numberOfSteps), itsLastFetchTime()
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

	itsLogger = std::unique_ptr<logger>(logger_factory::Instance()->GetLog("lagged_ensemble"));

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
	std::vector<int> missingForecasts(itsNumberOfSteps, 0);  // for each step
	std::vector<int> loadedForecasts(itsNumberOfSteps, 0);   // for each step
	forecast_time ftime(time);

	// Ordering: most recent forecast will be the last one in itsForecasts
	if (itsLastFetchTime.OriginDateTime().Empty())
	{
		itsLogger->Info("Fetching for the first timestep, lagged steps not included");
		const size_t currentStep = 0;
		for (const auto& desired : itsDesiredForecasts)
		{
			try
			{
				auto Info = f->Fetch(config, ftime, forecastLevel, itsParam, desired, false);
				itsForecasts.push_back(Info);
				loadedForecasts[currentStep]++;
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
					missingForecasts[currentStep]++;
				}
			}
		}
		itsLastFetchTime = ftime;
	}
	else
	{
		itsLogger->Info("Fetching for all timesteps");

		// Start from the 'earliest' timestep
		for (int currentStep = static_cast<int>(itsNumberOfSteps) - 1; currentStep >= 0; currentStep--)
		{
			ftime.OriginDateTime().Adjust(itsLagResolution, lag * currentStep);

			for (const auto& desired : itsDesiredForecasts)
			{
				try
				{
					auto Info = f->Fetch(config, ftime, forecastLevel, itsParam, desired, false);
					itsForecasts.push_back(Info);
					loadedForecasts[currentStep]++;
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
						missingForecasts[currentStep]++;
					}
				}
			}
			itsLastFetchTime = ftime;
		}
	}

	for (auto&& tupl : zip_range(loadedForecasts, missingForecasts))
	{
		VerifyValidForecastCount(tupl.get<0>(), tupl.get<1>());
	}
}

void lagged_ensemble::VerifyValidForecastCount(int numLoadedForecasts, int numMissingForecasts)
{
	if (itsMaximumMissingForecasts > 0)
	{
		if (numMissingForecasts >= itsMaximumMissingForecasts)
		{
			itsLogger->Fatal("maximum number of missing fields " + std::to_string(numMissingForecasts) + "/" +
			                 std::to_string(itsMaximumMissingForecasts) + " reached, aborting");
			exit(1);
		}
	}
	else
	{
		if (numMissingForecasts > 0)
		{
			itsLogger->Fatal("missing " + std::to_string(numMissingForecasts) + " of " +
			                 std::to_string(itsMaximumMissingForecasts) + " allowed missing fields of data");
			exit(1);
		}
	}
	itsLogger->Info("succesfully loaded " + std::to_string(numLoadedForecasts) + "/" +
	                std::to_string(itsDesiredForecasts.size()) + " fields");
}

HPTimeResolution lagged_ensemble::LagResolution() const { return itsLagResolution; }
int lagged_ensemble::Lag() const { return itsLag; }
size_t lagged_ensemble::NumberOfSteps() const { return itsNumberOfSteps; }
}  // namespace himan
