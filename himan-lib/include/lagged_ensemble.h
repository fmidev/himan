#ifndef LAGGED_ENSEMBLE_H
#define LAGGED_ENSEMBLE_H

#include "ensemble.h"
#include "time_ensemble.h"

namespace himan
{
class lagged_ensemble : public ensemble
{
   public:
	lagged_ensemble(const param& parameter, size_t ensembleSize, const time_duration& theLag, size_t numberOfSteps);

	void Fetch(std::shared_ptr<const plugin_configuration> config, const forecast_time& time,
	           const level& forecastLevel) override;

	/// @brief Returns the amount of lag
	time_duration Lag() const;

	/// @brief Returns the number of lagged steps in the ensemble
	size_t NumberOfSteps() const;

	/// @brief Verify and report the number of forecasts succesfully loaded.
	/// Abort execution if the specified limit is exceeded.
	void VerifyValidForecastCount(int numLoadedForecasts, int numMissingForecasts);

   private:
	time_duration itsLag;
	size_t itsNumberOfSteps;
	forecast_time itsLastFetchTime;
};
}  // namespace himan

#endif /* LAGGED_ENSEMBLE_H */
