//
// @file ensemble.h
//
//

#ifndef ENSEMBLE_H
#define ENSEMBLE_H

#include "forecast_time.h"
#include "himan_common.h"
#include "info.h"
#include "plugin_configuration.h"

namespace himan
{
enum HPEnsembleType
{
	kUnknownEnsembleType = 0,
	kPerturbedEnsemble,
	kTimeEnsemble,
	kLevelEnsemble,
	kLaggedEnsemble
};

const boost::unordered_map<HPEnsembleType, std::string> HPEnsembleTypeToString =
    ba::map_list_of(kUnknownEnsembleType, "unknown")(kPerturbedEnsemble, "perturbed ensemble")(
        kTimeEnsemble, "time ensemble")(kLevelEnsemble, "level ensemble")(kLaggedEnsemble, "lagged ensemble");

const boost::unordered_map<std::string, HPEnsembleType> HPStringToEnsembleType =
    ba::map_list_of("unknown", kUnknownEnsembleType)("perturbed ensemble", kPerturbedEnsemble)(
        "time ensemble", kTimeEnsemble)("level ensemble", kLevelEnsemble)("lagged ensemble", kLaggedEnsemble);

// ensemble is a thin layer on top of the usual himan data utilities.
// It is used to make working with ensemble forecasts a bit nicer and
// feel more like working with deterministic forecasts

class ensemble
{
   public:
	/// @brief Constructs an ensemble with one control forecast and expectedEnsembleSize - 1 perturbations
	ensemble(const param& parameter, size_t expectedEnsembleSize, int maximumMissingForecasts = 0);

	/// @brief Constructs an ensemble with control forecasts taken from `controlForecasts` and
	/// expectedEnsembleSize - controlForecasts.size() perturbations
	ensemble(const param& parameter, const std::vector<forecast_type>& desiredForecasts,
	         int maximumMissingForecasts = 0);

	virtual ~ensemble() = default;

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
	float Value(size_t forecastIndex) const;

	/// @brief Returns the current values of the ensemble
	std::vector<float> Values() const;

	/// @brief Returns the current values of the ensemble sorted in increasing order, missing values are removed
	std::vector<float> SortedValues() const;

	/// @brief Returns the mean value of the ensemble
	float Mean() const;

	/// @brief Returns the variance of the ensemble
	float Variance() const;

	/// @brief Returns Nth central moment of the ensemble
	float CentralMoment(int N) const;

	/// @brief Returns the size of the currently fetched ensemble
	size_t Size() const;

	virtual std::string ClassName() const;

	/// @brief Returns the expected size of the ensemble. NOTE: this can
	/// differ from the actual size of the ensemble!
	size_t ExpectedSize() const;

	param Param() const;
	void Param(const param& par);

	HPEnsembleType EnsembleType() const;

	int MaximumMissingForecasts() const;

	/// @brief Return all data for given ensemble member
	std::shared_ptr<info<float>> Forecast(size_t i);

	virtual std::vector<forecast_type> DesiredForecasts() const;

   protected:
	ensemble() = default;

	/// @brief Verifies that we have the required number of valid forecasts, else abort execution.
	/// Outputs diagnostics.
	virtual void VerifyValidForecastCount(int numMissingForecasts);

	/// @brief The parameter of the ensemble
	param itsParam;

	size_t itsExpectedEnsembleSize;

	/// @brief Initialized forecast_types used by Fetch()
	std::vector<forecast_type> itsDesiredForecasts;

	/// @brief Forecasts acquired with Fetch(), each call of Fetch() will overwrite the previous results
	std::vector<std::shared_ptr<info<float>>> itsForecasts;

	HPEnsembleType itsEnsembleType;

	logger itsLogger;

	/// @brief When Fetching(), this is the maximum number of missing forecasts we can tolerate.
	int itsMaximumMissingForecasts;
};

inline float ensemble::Value(size_t forecastIndex) const
{
	return itsForecasts[forecastIndex]->Value();
}
inline std::string ensemble::ClassName() const
{
	return "himan::ensemble";
}
inline param ensemble::Param() const
{
	return itsParam;
}
inline int ensemble::MaximumMissingForecasts() const
{
	return itsMaximumMissingForecasts;
}
}  // namespace himan

// ENSEMBLE_H
#endif
