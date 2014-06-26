/*
 * rain_type.h
 *
 *  Created on: Apr 10, 2012
 *      Author: partio, perämäki
 */

#ifndef RAIN_TYPE_H
#define RAIN_TYPE_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{

/**
 * @class rain_type
 *
 * @brief Calculate ...
 *
 */

class rain_type : public compiled_plugin, private compiled_plugin_base
{
public:
    rain_type();

    inline virtual ~rain_type() {}

    rain_type(const rain_type& other) = delete;
    rain_type& operator=(const rain_type& other) = delete;

    virtual void Process(std::shared_ptr<const plugin_configuration> conf);

    virtual std::string ClassName() const
    {
        return "himan::plugin::rain_type";
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

};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<rain_type> (new rain_type());
}

} // namespace plugin
} // namespace himan

#endif /* RAIN_TYPE_H */
