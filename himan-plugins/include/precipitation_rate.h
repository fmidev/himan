/*
 * precipitation_rate.h
 *
 */

#ifndef PRECIPITATION_RATE_H
#define PRECIPITATION_RATE_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"
#include <math.h>  // pow function

namespace himan
{
namespace plugin
{
/**
 * @class instant_precipitation
 *
 * @brief Calculates the instant precipitation rate for rain and solid precipitation by the heuristic formuale
 *         1. Rain_rate = (rho*WMR*1000/0.072)^(1/0.880)
 *         2. SPre_rate = (rho*SMR*1000/0.250)^(1/0.900)
 *        where rho is density, WMR is mixing ratio for water and SMR is mixing ratio for solid precipitation.
 *
 *        In case of negative mixing ratios the value is corrected to 0.0 kg/kg.
 *
 */

class precipitation_rate : public compiled_plugin, private compiled_plugin_base
{
   public:
	precipitation_rate();

	inline virtual ~precipitation_rate()
	{
	}
	precipitation_rate(const precipitation_rate& other) = delete;
	precipitation_rate& operator=(const precipitation_rate& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const
	{
		return "himan::plugin::precipitation_rate";
	}
	virtual HPPluginClass PluginClass() const
	{
		return kCompiled;
	}

   private:
	virtual void Calculate(std::shared_ptr<info<float>> theTargetInfo, unsigned short theThreadIndex);
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<precipitation_rate>();
}

}  // namespace plugin
}  // namespace himan

#endif /* PRECIPITATION_RATE_H */
