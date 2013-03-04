/**
 * @file dewpoint.h
 *
 * @date Jan 21, 2012
 * @author: partio
 */

#ifndef DEWPOINT_H
#define DEWPOINT_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{

/**
 * @class dewpoint
 *
 * @brief Calculate dewpoint from T and RH
 *
 * Source: journals.ametsoc.org/doi/pdf/10.1175/BAMS-86-2-225
 */

class dewpoint : public compiled_plugin, private compiled_plugin_base
{
public:
    dewpoint();

    inline virtual ~dewpoint() {}

    dewpoint(const dewpoint& other) = delete;
    dewpoint& operator=(const dewpoint& other) = delete;

    virtual void Process(std::shared_ptr<const plugin_configuration> conf);

    virtual std::string ClassName() const
    {
        return "himan::plugin::dewpoint";
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

    void Run(std::shared_ptr<info>, const std::shared_ptr<const configuration>& conf, unsigned short threadIndex);
    void Calculate(std::shared_ptr<info> myTargetInfo, const std::shared_ptr<const configuration>& conf, unsigned short threadIndex);

    bool itsUseCuda;
	
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<dewpoint> (new dewpoint());
}

} // namespace plugin
} // namespace himan


#endif /* DEWPOINT_H */
