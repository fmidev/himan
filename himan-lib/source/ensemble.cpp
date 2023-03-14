#include "ensemble.h"
#include "plugin_factory.h"

#include "numerical_functions.h"
#include <numeric>
#include <stddef.h>
#include <stdint.h>

#define HIMAN_AUXILIARY_INCLUDE
#include "fetcher.h"
#undef HIMAN_AUXILIARY_INCLUDE

namespace
{
std::vector<float> RemoveMissingValues(const std::vector<float>& vec)
{
	std::vector<float> ret;
	ret.reserve(vec.size());

	for (const auto& v : vec)
	{
		if (himan::IsValid(v))
		{
			ret.emplace_back(v);
		}
	}
	return ret;
}
}  // namespace

namespace himan
{
ensemble::ensemble(const param& parameter, size_t expectedEnsembleSize, int maximumMissingForecasts)
    : itsParam(parameter),
      itsExpectedEnsembleSize(expectedEnsembleSize),
      itsForecasts(),
      itsEnsembleType(kPerturbedEnsemble),
      itsLogger(logger("ensemble")),
      itsMaximumMissingForecasts(maximumMissingForecasts)
{
	itsDesiredForecasts.reserve(expectedEnsembleSize);
	itsDesiredForecasts.push_back(forecast_type(kEpsControl, 0));

	for (size_t i = 1; i < itsDesiredForecasts.capacity(); i++)
	{
		itsDesiredForecasts.push_back(forecast_type(kEpsPerturbation, static_cast<float>(i)));
	}
}

ensemble::ensemble(const param& parameter, size_t expectedEnsembleSize,
                   const std::vector<forecast_type>& controlForecasts, int maximumMissingForecasts)
    : itsParam(parameter),
      itsExpectedEnsembleSize(expectedEnsembleSize),
      itsForecasts(),
      itsEnsembleType(kPerturbedEnsemble),
      itsLogger(logger("ensemble")),
      itsMaximumMissingForecasts(maximumMissingForecasts)
{
	ASSERT(controlForecasts.size() < expectedEnsembleSize);

	itsDesiredForecasts.reserve(expectedEnsembleSize);

	for (const auto& c : controlForecasts)
	{
		itsDesiredForecasts.push_back(c);
	}

	for (size_t i = controlForecasts.size(); i < itsDesiredForecasts.capacity(); i++)
	{
		itsDesiredForecasts.push_back(forecast_type(kEpsPerturbation, static_cast<float>(i)));
	}
}

ensemble::ensemble(const ensemble& other)
    : itsParam(other.itsParam),
      itsExpectedEnsembleSize(other.itsExpectedEnsembleSize),
      itsDesiredForecasts(other.itsDesiredForecasts),
      itsForecasts(other.itsForecasts),
      itsEnsembleType(other.itsEnsembleType),
      itsLogger(logger("ensemble")),
      itsMaximumMissingForecasts(other.itsMaximumMissingForecasts)
{
}

ensemble& ensemble::operator=(const ensemble& other)
{
	itsParam = other.itsParam;
	itsExpectedEnsembleSize = other.itsExpectedEnsembleSize;
	itsDesiredForecasts = other.itsDesiredForecasts;
	itsForecasts = other.itsForecasts;
	itsEnsembleType = other.itsEnsembleType;
	itsMaximumMissingForecasts = other.itsMaximumMissingForecasts;

	itsLogger = logger("ensemble");

	return *this;
}

void ensemble::Fetch(std::shared_ptr<const plugin_configuration> config, const forecast_time& time,
                     const level& forecastLevel)
{
	auto f = GET_PLUGIN(fetcher);

	// We need to clear the forecasts vector every time we fetch to support ensembles
	// with a rotating member scheme (GLAMEPS). This means that the index of the forecast
	// doesn't reflect its position in the ensemble.
	itsForecasts.clear();

	int numMissingForecasts = 0;

	for (const auto& desired : itsDesiredForecasts)
	{
		try
		{
			auto info = f->Fetch<float>(config, time, forecastLevel, itsParam, desired, false);
			itsForecasts.push_back(info);
		}
		catch (HPExceptionType& e)
		{
			if (e != kFileDataNotFound && e != kFileMetaDataNotFound)
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

void ensemble::VerifyValidForecastCount(int numMissingForecasts)
{
	// This is for data that might not have all the fields for every timestep
	if (itsMaximumMissingForecasts > 0)
	{
		if (numMissingForecasts >= itsMaximumMissingForecasts)
		{
			itsLogger.Error(fmt::format("Maximum number of missing fields ({}) reached", itsMaximumMissingForecasts));
			throw kFileDataNotFound;
		}
	}
	// Normally, we don't except any of the fields to be missing, but at this point
	// we've already catched the exceptions
	else
	{
		if (numMissingForecasts > 0)
		{
			itsLogger.Error(fmt::format("Missing {} of {} allowed missing fields", numMissingForecasts,
			                            itsMaximumMissingForecasts));
			throw kFileDataNotFound;
		}
	}

	itsLogger.Info(fmt::format("Succesfully loaded {}/{} fields", itsForecasts.size(), itsDesiredForecasts.size()));
}

void ensemble::ResetLocation()
{
	for (auto& f : itsForecasts)
	{
		f->ResetLocation();
	}
}

bool ensemble::FirstLocation()
{
	for (auto& f : itsForecasts)
	{
		if (!f->FirstLocation())
		{
			return false;
		}
	}
	return true;
}

bool ensemble::NextLocation()
{
	for (auto& f : itsForecasts)
	{
		if (!f->NextLocation())
		{
			return false;
		}
	}
	return true;
}

std::vector<float> ensemble::Values() const
{
	std::vector<float> ret;
	ret.reserve(Size());

	std::for_each(itsForecasts.begin(), itsForecasts.end(),
	              [&](const std::shared_ptr<info<float>>& Info) { ret.emplace_back(Info->Value()); });

	return ret;
}

std::vector<float> ensemble::SortedValues() const
{
	std::vector<float> v = RemoveMissingValues(Values());
	std::sort(v.begin(), v.end());
	return v;
}

float ensemble::Mean() const
{
	return numerical_functions::Mean<float>(RemoveMissingValues(Values()));
}

float ensemble::Variance() const
{
	return numerical_functions::Variance<float>(RemoveMissingValues(Values()));
}

float ensemble::CentralMoment(int N) const
{
	std::vector<float> v = RemoveMissingValues(Values());
	float mu = Mean();
	std::for_each(v.begin(), v.end(), [=](float& d) { d = powf(d - mu, static_cast<float>(N)); });
	return numerical_functions::Mean<float>(v);
}

HPEnsembleType ensemble::EnsembleType() const
{
	return itsEnsembleType;
}
size_t ensemble::Size() const
{
	return itsForecasts.size();
}
size_t ensemble::ExpectedSize() const
{
	return itsExpectedEnsembleSize;
}
std::shared_ptr<info<float>> ensemble::Forecast(size_t i)
{
	if (itsForecasts.size() <= i)
	{
		throw kFileDataNotFound;
	}

	return itsForecasts.at(i);
}
void ensemble::Param(const param& par)
{
	itsParam = par;
}
std::vector<forecast_type> ensemble::DesiredForecasts() const
{
	return itsDesiredForecasts;
}
}  // namespace himan
