//
// @file ensemble.cpp
//
//

#include "ensemble.h"
#include "logger_factory.h"
#include "plugin_factory.h"

#include <stddef.h>
#include <stdint.h>

#define HIMAN_AUXILIARY_INCLUDE
#include "fetcher.h"
#undef HIMAN_AUXILIARY_INCLUDE

namespace himan
{
ensemble::ensemble(const param& parameter, size_t ensembleSize)
    : itsParam(parameter),
      itsEnsembleSize(ensembleSize)  // ensembleSize includes the control forecast
      ,
      itsPerturbations(std::vector<forecast_type>(ensembleSize - 1)),
      itsForecasts(std::vector<info_t>(ensembleSize)),
      itsEnsembleType(kPerturbedEnsemble),
      itsMaxMissing(0)
{
	int perturbationNumber = 1;
	for (auto& p : itsPerturbations)
	{
		p = forecast_type(kEpsPerturbation, static_cast<double>(perturbationNumber));
		perturbationNumber++;
	}

	itsLogger = std::unique_ptr<logger>(logger_factory::Instance()->GetLog("ensemble"));
}

ensemble::ensemble() : itsEnsembleType(kPerturbedEnsemble), itsMaxMissing(0)
{
	itsLogger = std::unique_ptr<logger>(logger_factory::Instance()->GetLog("ensemble"));
}

ensemble::~ensemble() {}
ensemble::ensemble(const ensemble& other)
    : itsParam(other.itsParam),
      itsEnsembleSize(other.itsEnsembleSize),
      itsPerturbations(other.itsPerturbations),
      itsForecasts(other.itsForecasts),
      itsEnsembleType(other.itsEnsembleType),
      itsMaxMissing(other.itsMaxMissing)
{
	itsLogger = std::unique_ptr<logger>(logger_factory::Instance()->GetLog("ensemble"));
}

ensemble& ensemble::operator=(const ensemble& other)
{
	itsParam = other.itsParam;
	itsEnsembleSize = other.itsEnsembleSize;
	itsPerturbations = other.itsPerturbations;
	itsForecasts = other.itsForecasts;
	itsEnsembleType = other.itsEnsembleType;
	itsMaxMissing = other.itsMaxMissing;

	itsLogger = std::unique_ptr<logger>(logger_factory::Instance()->GetLog("ensemble"));

	return *this;
}

void ensemble::Fetch(std::shared_ptr<const plugin_configuration> config, const forecast_time& time,
                     const level& forecastLevel)
{
	auto f = GET_PLUGIN(fetcher);

	// The vector needs to be resized so that we can still
	// a) push_back new forecasts 'densely'
	// b) loop through the forecasts with a range-based for without checking for 'missing forecasts'
	// This means that itsForecasts[2] isn't necessarily the perturbation with forecast_type_value 2.
	itsForecasts.resize(0);

	int numMissing = 0;

	// First get the control forecast
	try
	{
		auto info  = f->Fetch(config, time, forecastLevel, itsParam, forecast_type(kEpsControl, 0), false);
		itsForecasts.push_back(info);
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
			numMissing++;
		}
	}

	// Then get the perturbations
	for (const auto& perturbation : itsPerturbations)
	{
		try
		{
			auto info = f->Fetch(config, time, forecastLevel, itsParam, perturbation, false);
			itsForecasts.push_back(info);
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
				numMissing++;
			}
		}
	}

	// This is for data that might not have all the fields for every timestep
	if (itsMaxMissing > 0)
	{
		if (numMissing >= itsMaxMissing)
		{
			itsLogger->Fatal("maximum number of missing fields (" + std::to_string(itsMaxMissing) +
			                 ") reached, aborting");
			exit(1);
		}
	}
	// Normally, we don't except any of the fields to be missing, but at this point
	// we've already catched the exceptions
	else
	{
		if (numMissing > 0)
		{
			itsLogger->Fatal("missing " + std::to_string(numMissing) + " of " + std::to_string(itsMaxMissing) +
			                 " fields of data");
			exit(1);
		}
	}
}

void ensemble::ResetLocation()
{
	for (auto& f : itsForecasts)
	{
		f->ResetLocation();
	}
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
	std::vector<double> ret(itsForecasts.size());
	size_t i = 0;
	for (auto& f : itsForecasts)
	{
		ret[i] = f->Value();
		i++;
	}
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
	return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

HPEnsembleType ensemble::EnsembleType() const { return itsEnsembleType; }

size_t ensemble::Size() const { return itsForecasts.size(); }

}  // namespace himan
