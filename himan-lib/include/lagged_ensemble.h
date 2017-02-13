#ifndef LAGGED_ENSEMBLE_H
#define LAGGED_ENSEMBLE_H

#include "ensemble.h"

namespace himan
{
class lagged_ensemble : public ensemble
{
   public:
	lagged_ensemble(const param& parameter, size_t ensembleSize, HPTimeResolution lagResolution, int lag,
	                size_t numberOfSteps);

	void Fetch(std::shared_ptr<const plugin_configuration> config, const forecast_time& time,
	           const level& forecastLevel) override;

	/// @brief Returns the lag time resolution
	HPTimeResolution LagResolution() const;

	/// @brief Returns the amount of lag in LagResolution() units
	int Lag() const;

	/// @brief Returns the number of lagged steps in the ensemble
	size_t NumberOfSteps() const;

	/// @brief Verify and report the number of forecasts succesfully loaded.
	/// Abort execution if the specified limit is exceeded.
	void VerifyValidForecastCount(int numLoadedForecasts, int numMissingForecasts);

   private:
	HPTimeResolution itsLagResolution;
	int itsLag;
	size_t itsNumberOfSteps;
	forecast_time itsLastFetchTime;
};
}  // namespace himan

#endif /* LAGGED_ENSEMBLE_H */
