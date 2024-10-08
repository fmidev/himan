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

namespace
{
std::vector<std::pair<forecast_type, time_duration>> CreateNamedEnsembleConfiguration(const std::string& name)
{
	// MEPS member distribution as of 2020-02-28:
	// Operational member distribution
	// Time (UTC) 	CIRRUS 	STRATUS VOIMA
	// 00,03,…,21 	0 	1,2,12 	9
	// 01,04,…,22 	7 	3,4,13 	10
	// 02,05,…,23 	8,14 	5,6 	11
	//
	// Here we define a single MEPS ensemble to consist
	// of the control member cycle and two cycles *preceding it*.
	//
	// So ensemble where control member is produced at 00 cycle
	// include cycles 23 and 22.

	// TODO: think if this configuration could be stored outside code,
	// for example in database

	// clang-format off

	const static std::vector<std::pair<forecast_type, time_duration>> MEPS_MEMBER_CONFIGURATION({
	     {forecast_type(kEpsControl, 0), time_duration("00:00:00")},
	     {forecast_type(kEpsPerturbation, 1), time_duration("00:00:00")},
	     {forecast_type(kEpsPerturbation, 2), time_duration("00:00:00")},
	     {forecast_type(kEpsPerturbation, 3), time_duration("-02:00:00")},
	     {forecast_type(kEpsPerturbation, 4), time_duration("-02:00:00")},
	     {forecast_type(kEpsPerturbation, 5), time_duration("-01:00:00")},
	     {forecast_type(kEpsPerturbation, 6), time_duration("-01:00:00")},
	     {forecast_type(kEpsPerturbation, 7), time_duration("-02:00:00")},
	     {forecast_type(kEpsPerturbation, 8), time_duration("-01:00:00")},
	     {forecast_type(kEpsPerturbation, 9), time_duration("00:00:00")},
	     {forecast_type(kEpsPerturbation, 10), time_duration("-02:00:00")},
	     {forecast_type(kEpsPerturbation, 11), time_duration("-01:00:00")},
	     {forecast_type(kEpsPerturbation, 12), time_duration("00:00:00")},
	     {forecast_type(kEpsPerturbation, 13), time_duration("-02:00:00")},
	     {forecast_type(kEpsPerturbation, 14), time_duration("-01:00:00")}
	});

	// clang-format on

	if (name == "MEPS_SINGLE_ENSEMBLE")
	{
		return MEPS_MEMBER_CONFIGURATION;
	}
	else if (name == "MEPS_LAGGED_ENSEMBLE")
	{
		auto config = MEPS_MEMBER_CONFIGURATION;

		for (const auto& p : MEPS_MEMBER_CONFIGURATION)
		{
			config.push_back({p.first, p.second - THREE_HOURS});
		}

		return config;
	}

	throw std::runtime_error(fmt::format(
	    "Unable to create named ensemble for {}, allowed values are: MEPS_SINGLE_ENSEMBLE,MEPS_LAGGED_ENSEMBLE", name));
}
}  // namespace

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
	itsDesiredForecasts = CreateNamedEnsembleConfiguration(namedEnsemble);
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
