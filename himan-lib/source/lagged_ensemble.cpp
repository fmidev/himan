#include "lagged_ensemble.h"

#include "named_ensemble.h"
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
                                 size_t numberOfSteps, int maximumMissingForecasts)
    : lagged_ensemble(parameter, expectedEnsembleSize, theLag * static_cast<int>(numberOfSteps - 1), theLag * -1,
                      maximumMissingForecasts)
{
}

lagged_ensemble::lagged_ensemble(const param& parameter, size_t expectedEnsembleSize, const time_duration& theLag,
                                 const time_duration& theStep, int maximumMissingForecasts)
{
	itsLogger = logger("lagged_ensemble");

	if (theLag.Hours() > 0)
	{
		itsLogger.Fatal("Lag has to be negative");
		himan::Abort();
	}
	if (theStep.Hours() < 0)
	{
		itsLogger.Fatal(fmt::format("Step has to be positive ({})", theStep.Hours()));
		himan::Abort();
	}

	itsParam = parameter;
	itsEnsembleType = kLaggedEnsemble;

	itsDesiredForecasts.reserve(expectedEnsembleSize);

	time_duration currentLag("00:00:00");
	const time_duration end = theLag;

	while (currentLag >= end)
	{
		itsDesiredForecasts.push_back(std::make_pair(forecast_type(kEpsControl, 0), currentLag));

		for (size_t j = 1; j < expectedEnsembleSize; j++)
		{
			itsDesiredForecasts.push_back(
			    std::make_pair(forecast_type(kEpsPerturbation, static_cast<float>(j)), currentLag));
		}
		currentLag -= theStep;
	}

	itsForecasts.reserve(expectedEnsembleSize);
	itsMaximumMissingForecasts = maximumMissingForecasts;
}

lagged_ensemble::lagged_ensemble(const param& parameter,
                                 const std::vector<std::pair<forecast_type, time_duration>>& theConfiguration,
                                 int maximumMissingForecasts)
    : itsDesiredForecasts(theConfiguration)
{
	itsLogger = logger("lagged_ensemble");
	itsParam = parameter;
	itsEnsembleType = kLaggedEnsemble;
	itsForecasts.reserve(theConfiguration.size());
	itsMaximumMissingForecasts = maximumMissingForecasts;
}

lagged_ensemble::lagged_ensemble(const param& parameter, const std::string& namedEnsemble, int maximumMissingForecasts)
{
	itsLogger = logger("lagged_ensemble");
	itsParam = parameter;
	itsEnsembleType = kLaggedEnsemble;
	const auto& definition = GetNamedEnsemble(namedEnsemble);

	if (definition.type != kLaggedEnsemble)
	{
		itsLogger.Fatal(fmt::format("Named ensemble '{}' is not a lagged ensemble", namedEnsemble));
		himan::Abort();
	}

	itsDesiredForecasts = definition.members;
	itsForecasts.reserve(itsDesiredForecasts.size());
	itsMaximumMissingForecasts = maximumMissingForecasts;
}

lagged_ensemble::lagged_ensemble(const lagged_ensemble& other)
    : ensemble(other), itsDesiredForecasts(other.itsDesiredForecasts)
{
	itsEnsembleType = other.itsEnsembleType;
	itsLogger = logger("lagged_ensemble");
}

std::vector<forecast_type> lagged_ensemble::DesiredForecasts() const
{
	std::vector<forecast_type> vec;
	vec.reserve(itsDesiredForecasts.size());

	for (const auto& f : itsDesiredForecasts)
	{
		vec.push_back(f.first);
	}
	return vec;
}

void lagged_ensemble::Fetch(std::shared_ptr<const plugin_configuration> config, const forecast_time& time,
                            const level& forecastLevel)
{
	auto f = GET_PLUGIN(fetcher);

	itsForecasts.clear();

	int missing = 0;
	int loaded = 0;

	itsLogger.Info(fmt::format("Initial analysis time is {}", time.OriginDateTime().ToSQLTime()));

	for (const auto& p : itsDesiredForecasts)
	{
		const forecast_type& ftype = p.first;
		const time_duration& lag = p.second;

		forecast_time ftime(time);
		ftime.OriginDateTime() += lag;
		if (ftime.Step().Hours() < 0)
		{
			itsLogger.Trace("Negative leadtime, skipping");
			continue;
		}
		itsLogger.Trace(
		    fmt::format("Fetching {} with lag {}", static_cast<std::string>(ftype), static_cast<std::string>(lag)));

		try
		{
			auto Info = f->Fetch<float>(config, ftime, forecastLevel, itsParam, ftype, false);
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

	VerifyValidForecastCount(loaded, missing);
}

void lagged_ensemble::VerifyValidForecastCount(int numLoadedForecasts, int numMissingForecasts)
{
	if (itsMaximumMissingForecasts > 0)
	{
		if (numMissingForecasts > itsMaximumMissingForecasts)
		{
			itsLogger.Error(fmt::format("Maximum number of missing fields {}/{} reached", numMissingForecasts,
			                            itsMaximumMissingForecasts));
			throw kFileDataNotFound;
		}
	}
	else
	{
		if (numMissingForecasts > 0)
		{
			itsLogger.Error(fmt::format("Missing {} of {} allowed missing fields", numMissingForecasts,
			                            itsMaximumMissingForecasts));
			throw kFileDataNotFound;
		}
	}
	itsLogger.Info(fmt::format("Succesfully loaded {} fields", numLoadedForecasts));
}

}  // namespace himan
