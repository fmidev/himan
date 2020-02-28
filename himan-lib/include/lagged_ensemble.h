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
	lagged_ensemble(const param& parameter, size_t ensembleSize, const time_duration& theLag,
	                const time_duration& theStep = ONE_HOUR);
	lagged_ensemble(const param& parameter,
	                const std::vector<std::pair<forecast_type, time_duration>>& itsEnsembleConfiguration);
	lagged_ensemble(const param& parameter, const std::string& namedEnsemble);

	void Fetch(std::shared_ptr<const plugin_configuration> config, const forecast_time& time,
	           const level& forecastLevel) override;

	/// @brief Verify and report the number of forecasts succesfully loaded.
	/// Abort execution if the specified limit is exceeded.
	void VerifyValidForecastCount(int numLoadedForecasts, int numMissingForecasts);

   private:
	std::vector<std::pair<forecast_type, time_duration>> itsDesiredForecasts;
};

}  // namespace himan

#endif /* LAGGED_ENSEMBLE_H */
