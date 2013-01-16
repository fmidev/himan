/*
 * tk2tc.h
 *
 *  Created on: Nov 17, 2012
 *      Author: partio
 */

#ifndef ICING_H
#define ICING_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{

class icing : public compiled_plugin, private compiled_plugin_base
{
public:
    icing();

    inline virtual ~icing() {}

    icing(const icing& other) = delete;
    icing& operator=(const icing& other) = delete;

    virtual void Process(std::shared_ptr<configuration> theConfiguration);

    virtual std::string ClassName() const
    {
        return "himan::plugin::icing";
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

};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<icing> (new icing());
}

} // namespace plugin
} // namespace himan

#endif /* ICING */
