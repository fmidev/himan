/*
 * fog.h
 *
 *  Created on: Jul 5, 2012
 *      Author: perämäki
 */

#ifndef FOG_H
#define FOG_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{

/**
 * @class fog
 *
 * @brief Calculate ...
 *
 */

class fog : public compiled_plugin, private compiled_plugin_base
{
public:
    fog();

    inline virtual ~fog() {}

    fog(const fog& other) = delete;
    fog& operator=(const fog& other) = delete;

    virtual void Process(std::shared_ptr<const plugin_configuration> conf);

    virtual std::string ClassName() const
    {
        return "himan::plugin::fog";
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

    void Run(std::shared_ptr<info>, std::shared_ptr<const plugin_configuration> theConfiguration, unsigned short theThreadIndex);
    void Calculate(std::shared_ptr<info> theTargetInfo, std::shared_ptr<const plugin_configuration> theConfiguration, unsigned short theThreadIndex);

    bool itsUseCuda;
    int itsCudaDeviceCount;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<fog> (new fog());
}

} // namespace plugin
} // namespace himan

#endif /* FOG_H */
