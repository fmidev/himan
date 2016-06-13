//
// @file ensemble.cpp
//
// @date June 2, 2016
// @author vanhatam
//

#include "ensemble.h"
#include "fetcher.h"
#include "plugin_factory.h"

#include <stdint.h>
#include <stddef.h>

namespace himan
{

ensemble::ensemble(const param& parameter, size_t forecastCount)
	: m_param(parameter)
	, m_forecastCount(forecastCount)
{
	m_perturbations = std::vector<forecast_type> (forecastCount - 1); // forecast count includes the control forecast
	m_forecasts = std::vector<info_t> (forecastCount);

	int perturbationNumber = 1;
	for (auto & p : m_perturbations)
	{
		p = forecast_type (kEpsPerturbation, static_cast<double>(perturbationNumber));
		perturbationNumber++;
	}
}

ensemble::~ensemble()
{
}

void ensemble::Fetch(std::shared_ptr<const plugin_configuration> config, const forecast_time& time, const level& forecastLevel)
{
	// NOTE should this be stored some where else? Every time you call Fetch(), the instantiation will happen
	auto f = GET_PLUGIN(fetcher);

	try
	{
		// First get the control forecast
		m_forecasts[0] = f->Fetch(config, time, forecastLevel, m_param, forecast_type(kEpsControl, 0), false);

		// Then get the perturbations
		for (size_t i = 1; i < m_perturbations.size() + 1; i++)
		{
			m_forecasts[i] = f->Fetch(config, time, forecastLevel, m_param, m_perturbations[i-1], false);
		}
	}
	catch (HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw std::runtime_error("Ensemble: unable to proceed");
		} 
		else
		{
			// NOTE let the plugin decide what to do with missing data
			throw e;
		}
	}
}

void ensemble::ResetLocation()
{
	for (size_t i = 0; i < m_forecasts.size(); i++) 
	{
		assert(m_forecasts[i]);

		m_forecasts[i]->ResetLocation();
	}
}

bool ensemble::NextLocation()
{
	for (size_t i = 0; i < m_forecasts.size(); i++)
	{
		assert(m_forecasts[i]);

		if (!m_forecasts[i]->NextLocation())
		{
			return false;
		}
	}
	return true;
}

std::vector<double> ensemble::Values() const
{
	std::vector<double> ret (m_forecastCount);
	size_t i = 0;
	for (auto& f : m_forecasts)
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

} // namespace himan
