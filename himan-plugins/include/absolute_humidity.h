/*
 * absolute_humidity.h
 *
 */

#ifndef absolute_humidity_H
#define absolute_humidity_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"
#include <math.h>  // fmax function

namespace himan
{
namespace plugin
{
/**
 * @class instant_precipitation
 *
 * @brief Calculates the absolute humidity of air by the formula
 *          1. absolute_humidity = rho*(WMR+SMR)
 *        where rho is density, WMR is mixing ratio for water and SMR is mixing ratio for solid precipitation.
 *
 *        In case of negative mixing ratios the value is corrected to 0.0 kg/kg.
 *
 */

class absolute_humidity : public compiled_plugin, private compiled_plugin_base
{
   public:
	absolute_humidity();

	inline virtual ~absolute_humidity() {}
	absolute_humidity(const absolute_humidity& other) = delete;
	absolute_humidity& operator=(const absolute_humidity& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const { return "himan::plugin::absolute_humidity"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(0, 1); }
   private:
	virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<absolute_humidity>(new absolute_humidity());
}

}  // namespace plugin
}  // namespace himan

#endif /* absolute_humidity_H */
