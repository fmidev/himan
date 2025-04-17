/*
 * transformer.h
 *
 */

#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"
#include "ensemble.h"
#include "lagged_ensemble.h"
#include <vector>

namespace himan
{
namespace plugin
{
class transformer : public compiled_plugin, private compiled_plugin_base
{
   public:
	transformer();

	inline virtual ~transformer() = default;
	transformer(const transformer& other) = delete;
	transformer& operator=(const transformer& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::transformer";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kCompiled;
	}

   private:
	virtual void Calculate(std::shared_ptr<info<float>> theTargetInfo, unsigned short threadIndex) override;
	virtual void Calculate(std::shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex) override;

	// Check and write json parameters needed for transformer plug-in to local variables.
	void SetAdditionalParameters();
	std::vector<level> LevelsFromString(const std::string& levelType, const std::string& levelValues) const;
	void Rotate(std::shared_ptr<info<double>> myTargetInfo);
	std::shared_ptr<info<double>> InterpolateTime(const forecast_time& ftime, const level& lev, const param& par,
	                                              const forecast_type& ftype) const;
	std::shared_ptr<info<double>> InterpolateLevel(const forecast_time& ftime, const level& lev, const param& par,
	                                               const forecast_type& ftype) const;
	std::shared_ptr<info<double>> FetchSource(std::shared_ptr<info<double>>& myTargetInfo, forecast_time sourceTime,
	                                          forecast_type forecastType);

	double itsBase;
	double itsScale;
	std::vector<param> itsSourceParam;
	std::vector<param> itsTargetParam;
	std::vector<level> itsSourceLevels;
	bool itsApplyLandSeaMask;
	double itsLandSeaMaskThreshold;
	HPInterpolationMethod itsInterpolationMethod;
	forecast_type itsTargetForecastType;
	forecast_type itsSourceForecastType;
	bool itsRotateVectorComponents;
	bool itsDoTimeInterpolation;
	bool itsDoLevelInterpolation;
	std::string itsChangeMissingTo;
	bool itsWriteEmptyGrid;
	int itsDecimalPrecision;
	bool itsParamDefinitionFromConfig;
	std::unique_ptr<ensemble> itsEnsemble;
	time_duration itsSourceForecastPeriod;
	bool itsReadFromPreviousForecastIfNotFound;
	double itsMinimumValue;
	double itsMaximumValue;
	bool itsAllowAnySourceForecastType;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<transformer>(new transformer());
}
}  // namespace plugin
}  // namespace himan

#endif /* TRANSFORMER */
