/*
 * @file fmi_weather_symbol_1.h
 *
 * @date May, 2014
 * @author: Tack
 */

#ifndef FMI_WEATHER_SYMBOL_1_H
#define FMI_WEATHER_SYMBOL_1_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"
#include <boost/bimap.hpp>

namespace himan
{
namespace plugin
{

/**
 * @class fmi_weather_symbol_1
 *
 * @brief Calculate hessaa
 *
 */

class fmi_weather_symbol_1 : public compiled_plugin, private compiled_plugin_base
{
public:
    fmi_weather_symbol_1();

    inline virtual ~fmi_weather_symbol_1() {}

    fmi_weather_symbol_1(const fmi_weather_symbol_1& other) = delete;
    fmi_weather_symbol_1& operator=(const fmi_weather_symbol_1& other) = delete;

    virtual void Process(std::shared_ptr<const plugin_configuration> conf);

    virtual std::string ClassName() const
    {
        return "himan::plugin::fmi_weather_symbol_1";
    }

    virtual HPPluginClass PluginClass() const
    {
        return kCompiled;
    }

    virtual HPVersionNumber Version() const
    {
        return HPVersionNumber(1, 0);
    }

private:
	virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);
    double rain_type(double kIndex, double T0m, double T850);
    double thunder_prob(double kIndex, double cloud);
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<fmi_weather_symbol_1> (new fmi_weather_symbol_1());
}

} // namespace plugin
} // namespace himan

#endif /* fmi_weather_symbol_1 */
