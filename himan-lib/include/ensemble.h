//
// @file ensemble.h
//
//

#ifndef ENSEMBLE_H
#define ENSEMBLE_H

#include "forecast_time.h"
#include "himan_common.h"
#include "level.h"
#include "param.h"
#include "plugin_configuration.h"

namespace himan
{
// ensemble is a thin layer on top of the usual himan data utilities.
// It is used to make working with ensemble forecasts a bit nicer and
// feel more like working with deterministic forecasts

class ensemble
{
   public:
	ensemble(const param& parameter, size_t ensembleSize);

	ensemble();

	virtual ~ensemble();

	ensemble(const ensemble& other);

	ensemble& operator=(const ensemble& other);

	/// @brief Fetch the specified forecasts for the ensemble
	virtual void Fetch(std::shared_ptr<const plugin_configuration> config, const forecast_time& time,
	                   const level& forecastLevel);

	/// @brief Reset the location of all the ensembles
	void ResetLocation();

	/// @brief Increment the location of all the ensembles,
	/// if any of these fails, then all fail
	bool NextLocation();

	/// @brief Returns the current value of the specified forecast of the ensemble
	double Value(size_t forecastIndex) const;

	/// @brief Returns the current values of the ensemble
	std::vector<double> Values() const;

	/// @brief Returns the current values of the ensemble sorted in increasing order
	std::vector<double> SortedValues() const;

	/// @brief Returns the mean value of the ensemble
	double Mean() const;

	std::string ClassName() const;

	size_t Size() const;

	param Param() const;

   protected:
	/// @brief The parameter of the ensemble
	param itsParam;

	/// @brief The number of forecasts in this ensemble
	size_t itsEnsembleSize;

	/// @brief Initialized perturbations forecast_types used by Fetch()
	std::vector<forecast_type> itsPerturbations;

	/// @brief Forecasts acquired with Fetch(), each call of Fetch() will overwrite the previous results
	std::vector<info_t> itsForecasts;
};

inline size_t ensemble::Size() const { return itsEnsembleSize; }
inline double ensemble::Value(size_t forecastIndex) const { return itsForecasts[forecastIndex]->Value(); }
inline std::string ensemble::ClassName() const { return "himan::ensemble"; }
inline param ensemble::Param() const { return itsParam; }
}  // namespace himan

// ENSEMBLE_H
#endif
