/*
 * ncl.h
 *
 *  Created on: Apr 10, 2012
 *      Author: partio, perämäki
 */

#ifndef NCL_H
#define NCL_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{

/**
 * @class ncl
 *
 * @brief Calculate ...
 *
 */

class ncl : public compiled_plugin, private compiled_plugin_base
{
public:
    ncl();

    inline virtual ~ncl() {}

    ncl(const ncl& other) = delete;
    ncl& operator=(const ncl& other) = delete;

    virtual void Process(std::shared_ptr<const plugin_configuration> conf);

    virtual std::string ClassName() const
    {
        return "himan::plugin::ncl";
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
    std::shared_ptr<info> FetchPrevious(const forecast_time& wantedTime, const level& wantedLevel, const param& wantedParam);

    int targetTemperature;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<ncl> (new ncl());
}

} // namespace plugin
} // namespace himan

#endif /* NCL_H */
