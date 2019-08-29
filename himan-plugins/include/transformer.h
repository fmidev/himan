/*
 * transformer.h
 *
 */

#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"
#include <boost/property_tree/ptree.hpp>
#include <vector>

namespace himan
{
namespace plugin
{
class transformer : public compiled_plugin, private compiled_plugin_base
{
   public:
	transformer();

	inline virtual ~transformer()
	{
	}
	transformer(const transformer& other) = delete;
	transformer& operator=(const transformer& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const
	{
		return "himan::plugin::transformer";
	}
	virtual HPPluginClass PluginClass() const
	{
		return kCompiled;
	}

   protected:
	virtual void WriteToFile(const std::shared_ptr<info<double>> targetInfo,
	                         write_options opts = write_options()) override;

   private:
	virtual void Calculate(std::shared_ptr<info<double>> theTargetInfo, unsigned short theThreadIndex) override;

	// Check and write json parameters needed for transformer plug-in to local variables.
	void SetAdditionalParameters();
	std::vector<level> LevelsFromString(const std::string& levelType, const std::string& levelValues) const;
	void Rotate(himan::info_t myTargetInfo);
	std::shared_ptr<info<double>> InterpolateTime(const forecast_time& ftime, const level& lev, const param& par,
	                                              const forecast_type& ftype) const;

	double itsBase;
	double itsScale;
	std::vector<std::string> itsSourceParam;
	std::vector<std::string> itsTargetParam;
	std::vector<level> itsSourceLevels;
	bool itsApplyLandSeaMask;
	double itsLandSeaMaskThreshold;
	HPInterpolationMethod itsInterpolationMethod;
	forecast_type itsTargetForecastType;
	forecast_type itsSourceForecastType;
	bool itsRotateVectorComponents;
	bool itsDoTimeInterpolation;
	double itsChangeMissingTo;
	bool itsWriteEmptyGrid;
	int itsDecimalPrecision;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<transformer>(new transformer());
}
}  // namespace plugin
}  // namespace himan

#endif /* TRANSFORMER */
