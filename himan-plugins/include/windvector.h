/**
 * @file windvector.h
 *
 * @date Jan 21, 2013
 * @author aalto
 */

#ifndef WINDVECTOR_H
#define WINDVECTOR_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{

class windvector : public compiled_plugin, private compiled_plugin_base
{
public:
    windvector();

    inline virtual ~windvector() {}

    windvector(const windvector& other) = delete;
    windvector& operator=(const windvector& other) = delete;

    virtual void Process(std::shared_ptr<const plugin_configuration> conf);

    virtual std::string ClassName() const
    {
        return "himan::plugin::windvector";
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

    void Run(std::shared_ptr<info>, std::shared_ptr<const configuration> theConfiguration, unsigned short theThreadIndex);
    void Calculate(std::shared_ptr<info> theTargetInfo, std::shared_ptr<const configuration> theConfiguration, unsigned short theThreadIndex);

    bool itsUseCuda;
    bool itsSeaCalculation;
    bool itsIceCalculation;
    bool itsAirCalculation;

};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<windvector> (new windvector());
}

} // namespace plugin
} // namespace himan

#endif /* WINDVECTOR */
