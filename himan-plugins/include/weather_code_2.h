/*
 * @file weather_code_2.h
 *
 */

#ifndef weather_code_2_H
#define weather_code_2_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class weather_code_2
 *
 * @brief Calculate hessaa
 *
 */

class weather_code_2 : public compiled_plugin, private compiled_plugin_base
{
   public:
	weather_code_2();

	inline virtual ~weather_code_2()
	{
	}
	weather_code_2(const weather_code_2& other) = delete;
	weather_code_2& operator=(const weather_code_2& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::weather_code_2";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kCompiled;
	}

   private:
	virtual void Calculate(std::shared_ptr<info<double>> theTargetInfo, unsigned short theThreadIndex);
	double rain_type(double kIndex, double T0m, double T850);
	double thunder_prob(double kIndex, double cloud);
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<weather_code_2>(new weather_code_2());
}
}  // namespace plugin
}  // namespace himan

#endif /* weather_code_2 */
