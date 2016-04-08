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

#define HIMAN_AUXILIARY_INCLUDE

namespace himan
{
namespace plugin
{

class compiled_plugin : public himan_plugin
{
public:

    compiled_plugin() : itsCudaEnabledCalculation(false) {}

    virtual ~compiled_plugin() {}

    virtual void Process(std::shared_ptr<const plugin_configuration> configuration) = 0;

    virtual std::string Formula()
    {
        return itsClearTextFormula;
    }
    virtual void Formula(std::string theClearTextFormula)
    {
        itsClearTextFormula = theClearTextFormula;
    }

	bool CudaEnabledCalculation() const
	{
		return itsCudaEnabledCalculation;
	}

protected:

    std::string itsClearTextFormula;
	bool itsCudaEnabledCalculation;

};

} // namespace plugin
} // namespace himan

#endif /* COMPILED_PLUGIN_H */
