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
	time_ensemble(const param& parameter, int maximumMissingForecasts = 0);
	time_ensemble(const param& parameter, size_t primaryTimeMaskLen, HPTimeResolution primaryTimeSpan,
	              int secondaryTimeMaskLen, int secondaryTimeMaskStep, HPTimeResolution secondaryTimeSpan,
	              int maximumMissingForecasts = 0);

	void Fetch(std::shared_ptr<const plugin_configuration> config, const forecast_time& time,
	           const level& forecastLevel) override;

	HPTimeResolution PrimaryTimeSpan() const
	{
		return itsPrimaryTimeSpan;
	}
	int SecondaryTimeMaskLen() const
	{
		return itsSecondaryTimeMaskLen;
	}
	int SecondaryTimeMaskStep() const
	{
		return itsSecondaryTimeMaskStep;
	}
	HPTimeResolution SecondaryTimeSpan() const
	{
		return itsSecondaryTimeSpan;
	}
	virtual std::string ClassName() const final
	{
		return "himan::time_ensemble";
	}

   private:
	HPTimeResolution itsPrimaryTimeSpan;
	int itsSecondaryTimeMaskLen = 0;
	int itsSecondaryTimeMaskStep = 1;
	HPTimeResolution itsSecondaryTimeSpan = kHourResolution;
};
}  // namespace himan
#endif /* TIME_ENSEMBLE_H */
