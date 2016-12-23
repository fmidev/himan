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
	/// @brief Constructs an ensemble with one control forecast and expectedEnsembleSize - 1 perturbations
	ensemble(const param& parameter, size_t expectedEnsembleSize);

	/// @brief Constructs an ensemble with control forecasts taken from `controlForecasts` and
	/// expectedEnsembleSize - controlForecasts.size() perturbations
	ensemble(const param& parameter, size_t expectedEnsembleSize, const std::vector<forecast_type>& controlForecasts);

	ensemble();

	virtual ~ensemble();

	ensemble(const ensemble& other);

	ensemble& operator=(const ensemble& other);

	/// @brief Fetch the specified forecasts for the ensemble
	virtual void Fetch(std::shared_ptr<const plugin_configuration> config, const forecast_time& time,
	                   const level& forecastLevel);

	/// @brief Reset the location of all the ensembles
	void ResetLocation();

	/// @brief Set the location of all the ensembles to the first location.
	/// Returns true if all locations are set to first succesfully,
	/// otherwise returns false.
	bool FirstLocation();

	/// @brief Increment the location of all the ensembles.
	/// Returns true if all the locations are incremented,
	/// otherwise returns false.
	bool NextLocation();

	/// @brief Returns the current value of the specified forecast of the ensemble
	double Value(size_t forecastIndex) const;

	/// @brief Returns the current values of the ensemble
	std::vector<double> Values() const;

	/// @brief Returns the current values of the ensemble sorted in increasing order
	std::vector<double> SortedValues() const;

	/// @brief Returns the mean value of the ensemble
	double Mean() const;

	/// @brief Returns the variance of the ensemble
	double Variance() const;

	/// @brief Returns Nth central moment of the ensemble
	double CentralMoment(int N) const;

	/// @brief Returns the size of the currently fetched ensemble
	size_t Size() const;

	std::string ClassName() const;

	/// @brief Returns the expected size of the ensemble. NOTE: this can
	/// differ from the actual size of the ensemble!
	size_t ExpectedSize() const;

	param Param() const;

	HPEnsembleType EnsembleType() const;

	int MaximumMissingForecasts() const;

	void MaximumMissingForecasts(int maximumMissing);

   protected:
	/// @brief Verifies that we have the required number of valid forecasts, else abort execution.
	/// Outputs diagnostics.
	virtual void VerifyValidForecastCount(int numMissingForecasts);

	/// @brief The parameter of the ensemble
	param itsParam;

	size_t itsExpectedEnsembleSize;

	/// @brief Initialized forecast_types used by Fetch()
	std::vector<forecast_type> itsDesiredForecasts;

	/// @brief Forecasts acquired with Fetch(), each call of Fetch() will overwrite the previous results
	std::vector<info_t> itsForecasts;

	HPEnsembleType itsEnsembleType;

	std::unique_ptr<logger> itsLogger;

	/// @brief When Fetching(), this is the maximum number of missing forecasts we can tolerate.
	int itsMaximumMissingForecasts;
};

inline double ensemble::Value(size_t forecastIndex) const { return itsForecasts[forecastIndex]->Value(); }
inline std::string ensemble::ClassName() const { return "himan::ensemble"; }
inline param ensemble::Param() const { return itsParam; }
inline int ensemble::MaximumMissingForecasts() const { return itsMaximumMissingForecasts; }
inline void ensemble::MaximumMissingForecasts(int maximumMissing) { itsMaximumMissingForecasts = maximumMissing; }
}  // namespace himan

// ENSEMBLE_H
#endif
