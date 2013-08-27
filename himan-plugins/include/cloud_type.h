/*
 * cloud_type.h
 *
 *  Created on: Jun 13, 2012
 *      Author: perämäki
 */

#ifndef CLOUD_TYPE_H
#define CLOUD_TYPE_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{

class cloud_type : public compiled_plugin, private compiled_plugin_base
{
public:
    cloud_type();

    inline virtual ~cloud_type() {}

    cloud_type(const cloud_type& other) = delete;
    cloud_type& operator=(const cloud_type& other) = delete;

    virtual void Process(std::shared_ptr<const plugin_configuration> conf);

    virtual std::string ClassName() const
    {
        return "himan::plugin::cloud_type";
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
    int DoMatako(double T2m, double T850);

};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<cloud_type> (new cloud_type());
}

} // namespace plugin
} // namespace himan

#endif /* CLOUD_TYPE_H */
