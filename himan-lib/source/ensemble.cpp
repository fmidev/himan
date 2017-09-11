//
// @file ensemble.cpp
//
//

#include "ensemble.h"
#include "plugin_factory.h"

#include <numeric>
#include <stddef.h>
#include <stdint.h>

#define HIMAN_AUXILIARY_INCLUDE
#include "fetcher.h"
#undef HIMAN_AUXILIARY_INCLUDE

namespace himan
{
ensemble::ensemble(const param& parameter, size_t expectedEnsembleSize)
    : itsParam(parameter),
      itsExpectedEnsembleSize(expectedEnsembleSize),
      itsForecasts(),
      itsEnsembleType(kPerturbedEnsemble),
      itsLogger(logger("ensemble")),
      itsMaximumMissingForecasts(0)
{
	itsDesiredForecasts.reserve(expectedEnsembleSize);
	itsDesiredForecasts.push_back(forecast_type(kEpsControl, 0));

	for (size_t i = 1; i < itsDesiredForecasts.capacity(); i++)
	{
		itsDesiredForecasts.push_back(forecast_type(kEpsPerturbation, static_cast<double>(i)));
	}
}

ensemble::ensemble(const param& parameter, size_t expectedEnsembleSize,
                   const std::vector<forecast_type>& controlForecasts)
    : itsParam(parameter),
      itsExpectedEnsembleSize(expectedEnsembleSize),
      itsForecasts(),
      itsEnsembleType(kPerturbedEnsemble),
      itsLogger(logger("ensemble")),
      itsMaximumMissingForecasts(0)
{
	assert(controlForecasts.size() < expectedEnsembleSize);

	itsDesiredForecasts.reserve(expectedEnsembleSize);

	for (const auto& c : controlForecasts)
	{
		itsDesiredForecasts.push_back(c);
	}

	for (size_t i = controlForecasts.size(); i < itsDesiredForecasts.capacity(); i++)
	{
		itsDesiredForecasts.push_back(forecast_type(kEpsPerturbation, static_cast<double>(i)));
	}
}

ensemble::ensemble()
    : itsExpectedEnsembleSize(0),
      itsEnsembleType(kPerturbedEnsemble),
      itsLogger(logger("ensemble")),
      itsMaximumMissingForecasts(0)
{
}

ensemble::~ensemble() {}
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
			auto info = f->Fetch(config, time, forecastLevel, itsParam, desired, false);
			itsForecasts.push_back(info);
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
			itsLogger.Fatal("maximum number of missing fields (" + std::to_string(itsMaximumMissingForecasts) +
			                ") reached, aborting");
			abort();
		}
	}
	// Normally, we don't except any of the fields to be missing, but at this point
	// we've already catched the exceptions
	else
	{
		if (numMissingForecasts > 0)
		{
			itsLogger.Fatal("missing " + std::to_string(numMissingForecasts) + " of " +
			                std::to_string(itsMaximumMissingForecasts) + " allowed missing fields of data");
			throw kFileDataNotFound;
		}
	}

	itsLogger.Info("succesfully loaded " + std::to_string(itsForecasts.size()) + "/" +
	               std::to_string(itsDesiredForecasts.size()) + " fields");
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

std::vector<double> ensemble::Values() const
{
	std::vector<double> ret;
	ret.reserve(Size());

	// Clients of ensemble shouldn't worry about missing values
	std::for_each(itsForecasts.begin(), itsForecasts.end(), [&](const info_t& Info) {
		const double v = Info->Value();
		if (IsValid(v))
		{
			ret.push_back(v);
		}
	});

	return ret;
}

std::vector<double> ensemble::SortedValues() const
{
	std::vector<double> v = Values();
	std::sort(v.begin(), v.end());
	return v;
}

double ensemble::Mean() const
{
	std::vector<double> v = Values();
	if (v.size() == 0)
	{
		return MissingDouble();
	}

	return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

double ensemble::Variance() const
{
	std::vector<double> values = Values();
	if (values.size() == 0)
	{
		return MissingDouble();
	}

	const double mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());

	double sum = 0.0;
	for (const auto& x : values)
	{
		const double t = x - mean;
		sum += t * t;
	}

	return sum / static_cast<double>(values.size());
}

double ensemble::CentralMoment(int N) const
{
	std::vector<double> v = Values();
	double mu = Mean();
	std::for_each(v.begin(), v.end(), [=](double& d) { d = std::pow(d - mu, N); });
	return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

HPEnsembleType ensemble::EnsembleType() const { return itsEnsembleType; }
size_t ensemble::Size() const { return itsForecasts.size(); }
size_t ensemble::ExpectedSize() const { return itsExpectedEnsembleSize; }
info_t ensemble::Forecast(size_t i)
{
	if (itsForecasts.size() <= i)
	{
		throw kFileDataNotFound;
	}

	return itsForecasts[i];
}

}  // namespace himan
