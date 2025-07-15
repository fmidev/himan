#include "ensemble.h"
#include "plugin_factory.h"

#include "numerical_functions.h"
#include <numeric>
#include <stddef.h>
#include <stdint.h>
#include <util.h>

#define HIMAN_AUXILIARY_INCLUDE
#include "fetcher.h"
#undef HIMAN_AUXILIARY_INCLUDE

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

ensemble::ensemble(const param& parameter, const std::vector<forecast_type>& desiredForecasts,
                   int maximumMissingForecasts)
    : itsParam(parameter),
      itsDesiredForecasts(desiredForecasts),
      itsForecasts(),
      itsEnsembleType(kPerturbedEnsemble),
      itsLogger(logger("ensemble")),
      itsMaximumMissingForecasts(maximumMissingForecasts)
{
}

ensemble::ensemble(const param& parameter, const std::string& name, int maximumMissingForecasts)
    : itsParam(parameter),
      itsExpectedEnsembleSize(),
      itsForecasts(),
      itsEnsembleType(kPerturbedEnsemble),
      itsLogger(logger("ensemble")),
      itsMaximumMissingForecasts(maximumMissingForecasts)
{
	if (name == "ECMWF50")
	{
		itsExpectedEnsembleSize = 50;
		itsDesiredForecasts.reserve(itsExpectedEnsembleSize);

		for (size_t i = 1; i <= itsExpectedEnsembleSize; i++)
		{
			itsDesiredForecasts.push_back(forecast_type(kEpsPerturbation, static_cast<float>(i)));
		}
	}
	else if (name == "ECMWF51")
	{
		itsExpectedEnsembleSize = 51;
		itsDesiredForecasts.reserve(itsExpectedEnsembleSize);

		itsDesiredForecasts.push_back(forecast_type(kEpsControl, 0));
		for (size_t i = 1; i < itsExpectedEnsembleSize; i++)
		{
			itsDesiredForecasts.push_back(forecast_type(kEpsPerturbation, static_cast<float>(i)));
		}
	}
	else
	{
		itsLogger.Fatal(
		    fmt::format("Unable to create named ensemble for {}, allowed values are: ECMWF50,ECMWF51", name));
		himan::Abort();
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
	// Normally, we don't expect any of the fields to be missing, but at this point
	// we've already caught the exceptions
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
	return SortedValues(kRemove);
}

std::vector<float> ensemble::SortedValues(const HPMissingValueTreatment treatMissing) const
{
	std::vector<float> res;
	switch (treatMissing)
	{
		// The base case covers data where there usually are no missing values expected to be found in the
		// data or where they are not meaningful, e.g. in temperature fields where missing values should
		// be ignored and products like fractiles be computed from a reduced ensemble.
		case kRemove:
		{
			res = util::RemoveMissingValues(Values());
			std::sort(res.begin(), res.end());
			break;
		}
		// For some parameters missing values are used to indicate absence of the parameter, e.g. cloud
		// base height that is only meaningful when there are clouds. Cloud free cases are marked as
		// missing value. If we would remove missing values in this case the fractiles would always indicate
		// high probabilities for cloud base so we don't want to reduce the ensemble size here. Instead
		// we move the missing values to the end of the sorted ensemble so they would resemble infinite
		// cloude base height.
		case kLast:
		{
			std::vector<float> val = Values();
			std::vector<float> v = util::RemoveMissingValues(val);
			std::sort(v.begin(), v.end());

			res = std::vector<float>(val.size(), himan::MissingFloat());
			std::copy(v.begin(), v.end(), res.begin());
			break;
		}
		// This might be useful for similar reasons than kLast except that the missing value would be
		// representing infinite negative numbers in the sorting.
		case kFirst:
		{
			std::vector<float> val = Values();
			std::vector<float> v = util::RemoveMissingValues(val);
			std::sort(v.begin(), v.end());

			res = std::vector<float>(val.size(), himan::MissingFloat());
			std::copy(v.begin(), v.end(), std::back_inserter(res));
			break;
		}
		default:
		{
			itsLogger.Fatal("Undefined behaviour for missing value treatment");
			himan::Abort();
		}
	}
	return res;
}

float ensemble::Mean() const
{
	return numerical_functions::Mean<float>(util::RemoveMissingValues(Values()));
}

float ensemble::Variance() const
{
	return numerical_functions::Variance<float>(util::RemoveMissingValues(Values()));
}

float ensemble::CentralMoment(int N) const
{
	std::vector<float> v = util::RemoveMissingValues(Values());
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
