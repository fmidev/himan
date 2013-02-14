/**
 * @file compiled_plugin.h
 *
 * @brief Interface for compiled plugins
 *
 * @author partio
 */

#ifndef COMPILED_PLUGIN_H
#define COMPILED_PLUGIN_H

#include "himan_plugin.h"
#include "configuration.h"
#include "plugin_configuration.h"

#include <boost/thread.hpp>

namespace himan
{
namespace plugin
{

class compiled_plugin : public himan_plugin
{
public:

    compiled_plugin() {}

    virtual ~compiled_plugin() {}

    virtual void Process(std::shared_ptr<const configuration> configuration,
    						std::shared_ptr<info> targetInfo) = 0;

    virtual std::string Formula()
    {
        return itsClearTextFormula;
    }
    virtual void Formula(std::string theClearTextFormula)
    {
        itsClearTextFormula = theClearTextFormula;
    }

protected:

    std::string itsClearTextFormula;

};

} // namespace plugin
} // namespace himan

#endif /* COMPILED_PLUGIN_H */
