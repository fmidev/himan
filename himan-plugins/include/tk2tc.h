/*
 * tk2tc.h
 *
 *  Created on: Nov 17, 2012
 *      Author: partio
 */

#ifndef TK2TC_H
#define TK2TC_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{

class tk2tc : public compiled_plugin, private compiled_plugin_base
{
public:
    tk2tc();

    inline virtual ~tk2tc() {}

    tk2tc(const tk2tc& other) = delete;
    tk2tc& operator=(const tk2tc& other) = delete;

    virtual void Process(std::shared_ptr<const configuration> conf,
    						std::shared_ptr<info> targetInfo);

    virtual std::string ClassName() const
    {
        return "himan::plugin::tk2tc";
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
    return std::shared_ptr<tk2tc> (new tk2tc());
}

} // namespace plugin
} // namespace himan

#endif /* TK2TC */
