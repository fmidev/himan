//
// @file ensemble.h
//
// @date June 2, 2016
// @author vanhatam
//

#ifndef ENSEMBLE_H
#define ENSEMBLE_H

#include "himan_common.h"
#include "param.h"
#include "level.h"
#include "forecast_time.h"
#include "plugin_configuration.h"

namespace himan
{

// ensemble is a thin layer on top of the usual himan data utilities.
// It is used to make working with ensemble forecasts a bit nicer and
// feel more like working with deterministic forecasts

struct ensemble
{
	ensemble(const param& parameter, size_t forecastCount);
	~ensemble();

	// @brief Fetch the specified forecasts for the ensemble
	void Fetch(std::shared_ptr<const plugin_configuration> config, const forecast_time& time, const level& forecastLevel);

	// @brief Reset the location of all the ensembles
	void ResetLocation();

	// @brief Increment the location of all the ensembles,
	// if any of these fails, then all fail
	bool NextLocation();

	// @brief Returns the current value of the specified forecast of the ensemble
	double Value(size_t forecastIndex) const;

	// @brief Returns the current values of the ensemble
	std::vector<double> Values() const;

	// @brief Returns the current values of the ensemble sorted from in increasing order
	std::vector<double> SortedValues() const;

	std::string ClassName() const;

	size_t Size() const;

	// @brief The parameter of the ensemble
	param m_param;

	// @brief The number of forecasts in this ensemble
    size_t m_forecastCount;

	// @brief Initialized perturbations forecast_types used by Fetch()
    std::vector<forecast_type> m_perturbations;

	// @brief Forecasts acquired with Fetch(), each call of Fetch() will overwrite the previous results
	std::vector<info_t> m_forecasts;
};

inline size_t ensemble::Size() const
{
	return m_forecastCount;
}

inline double ensemble::Value(size_t forecastIndex) const
{
	return m_forecasts[forecastIndex]->Value();
}

inline std::string ensemble::ClassName() const
{
	return "himan::ensemble";
}


} // namespace himan

// ENSEMBLE_H
#endif
