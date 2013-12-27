/**
 * @file relative_humidity.h
 *
 * @date Jan Oct 7, 2013
 * @author: partio
 */

#ifndef RELATIVE_HUMIDITY_H
#define RELATIVE_HUMIDITY_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{

/**
 * @class relative_humidity
 *
 */

class relative_humidity : public compiled_plugin, private compiled_plugin_base
{
public:
    relative_humidity();

    inline virtual ~relative_humidity() {}

    relative_humidity(const relative_humidity& other) = delete;
    relative_humidity& operator=(const relative_humidity& other) = delete;

    virtual void Process(std::shared_ptr<const plugin_configuration> conf);

    virtual std::string ClassName() const
    {
        return "himan::plugin::relative_humidity";
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
	virtual void Calculate(std::shared_ptr<info> myTargetInfo, unsigned short threadIndex);

};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<relative_humidity> (new relative_humidity());
}

} // namespace plugin
} // namespace himan


#endif /* RELATIVE_HUMIDITY_H */
