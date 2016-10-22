//
// @file ensemble.cpp
//
//

#include "ensemble.h"
#include "plugin_factory.h"
#include "logger_factory.h"

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
      itsForecasts(std::vector<info_t>(ensembleSize))
{
	int perturbationNumber = 1;
	for (auto& p : itsPerturbations)
	{
		p = forecast_type(kEpsPerturbation, static_cast<double>(perturbationNumber));
		perturbationNumber++;
	}

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("time_ensemble"));
}

ensemble::ensemble() 
{
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("time_ensemble"));
}
ensemble::~ensemble() {}
ensemble::ensemble(const ensemble& other)
    : itsParam(other.itsParam),
      itsEnsembleSize(other.itsEnsembleSize),
      itsPerturbations(other.itsPerturbations),
      itsForecasts(other.itsForecasts)
{
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("time_ensemble"));
}

ensemble& ensemble::operator=(const ensemble& other)
{
	itsParam = other.itsParam;
	itsEnsembleSize = other.itsEnsembleSize;
	itsPerturbations = other.itsPerturbations;
	itsForecasts = other.itsForecasts;
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("time_ensemble"));

	return *this;
}

void ensemble::Fetch(std::shared_ptr<const plugin_configuration> config, const forecast_time& time,
                     const level& forecastLevel)
{
	// NOTE should this be stored some where else? Every time you call Fetch(), the instantiation will happen
	auto f = GET_PLUGIN(fetcher);

	try
	{
		// First get the control forecast
		itsForecasts[0] = f->Fetch(config, time, forecastLevel, itsParam, forecast_type(kEpsControl, 0), false);

		// Then get the perturbations
		for (size_t i = 1; i < itsPerturbations.size() + 1; i++)
		{
			itsForecasts[i] = f->Fetch(config, time, forecastLevel, itsParam, itsPerturbations[i - 1], false);
		}
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
			// NOTE let the plugin decide what to do with missing data
			throw;
		}
	}
}

void ensemble::ResetLocation()
{
	for (size_t i = 0; i < itsForecasts.size(); i++)
	{
		assert(itsForecasts[i]);

		itsForecasts[i]->ResetLocation();
	}
}

bool ensemble::NextLocation()
{
	for (size_t i = 0; i < itsForecasts.size(); i++)
	{
		assert(itsForecasts[i]);

		if (!itsForecasts[i]->NextLocation())
		{
			return false;
		}
	}
	return true;
}

std::vector<double> ensemble::Values() const
{
	std::vector<double> ret(itsEnsembleSize);
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

}  // namespace himan
