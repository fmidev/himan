/*
 * precipitation_rate.h
 *
 *  Created on: Mar 14, 2014
 *      Author: Tack
 */

#ifndef PRECIPITATION_RATE_H
#define PRECIPITATION_RATE_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"
#include <math.h> // pow function

namespace himan
{
namespace plugin
{

/**
 * @class instant_precipitation
 *
 * @brief Calculate ...
 *
 */

class precipitation_rate : public compiled_plugin, private compiled_plugin_base
{
public:
    precipitation_rate();

    inline virtual ~precipitation_rate() {}

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

    virtual HPVersionNumber Version() const
    {
        return HPVersionNumber(0, 1);
    }

private:
    virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);

};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<precipitation_rate> (new precipitation_rate());
}

} // namespace plugin
} // namespace himan

#endif /* PRECIPITATION_RATE_H */
