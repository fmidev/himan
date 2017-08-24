#ifndef POT_H
#define POT_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class time_series
 *
 * @brief contains a series of consecutive time instances for a given parameter field
 * This in preliminarily implemented with the POT plugin while the final implementation
 * will be part of himan-lib.
 */

class time_series
{
   public:
	time_series() = default;
	time_series(param theParam, size_t expectedSize);
	~time_series() = default;
	time_series(const time_series& other) = default;
	time_series& operator=(const time_series& other) = default;

	void Fetch(std::shared_ptr<const plugin_configuration> config, forecast_time start_time,
	           const HPTimeResolution& timeRes, int stepSize, int numSteps, const level& forecastLevel,
	           const forecast_type& requestedType, bool readPackedData);

	param Param() const { return itsParam; };
	void Param(param theParam);

	size_t Size() const { return itsInfos.size(); };
	std::vector<info_t>::iterator begin() { return itsInfos.begin(); };
	std::vector<info_t>::iterator end() { return itsInfos.end(); };
   private:
	param itsParam;
	std::vector<info_t> itsInfos;
};

/**
 * @class pot
 *
 * @brief Calculate the thunder probability
 *
 */

class pot : public compiled_plugin, private compiled_plugin_base
{
   public:
	pot();

	inline virtual ~pot() {}
	pot(const pot& other) = delete;
	pot& operator=(const pot& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const { return "himan::plugin::pot"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(0, 1); }
   private:
	virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);
	bool itsStrictMode;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::make_shared<pot>(); }
}  // namespace plugin
}  // namespace himan

#endif /* POT_PLUGIN_H */
