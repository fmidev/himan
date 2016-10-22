/*
 * File:   time_ensemble.h
 * Author: partio
 *
 * Created on October 20, 2016, 1:51 PM
 */

#ifndef TIME_ENSEMBLE_H
#define TIME_ENSEMBLE_H

#include "ensemble.h"

namespace himan
{
class time_ensemble : public ensemble
{
   public:
	time_ensemble(const param& parameter);
	time_ensemble(const param& parameter, size_t ensembleSize, HPTimeResolution timeSpan = kYearResolution);

	void Fetch(std::shared_ptr<const plugin_configuration> config, const forecast_time& time,
	           const level& forecastLevel) override;

   private:
	HPTimeResolution itsTimeSpan;
};
}
#endif /* TIME_ENSEMBLE_H */
