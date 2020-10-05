/*
 * density.h
 *
 */

#ifndef DENSITY_H
#define DENSITY_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class DENSITY
 *
 * @brief Calculates density from pressure and temperature using the ideal gas law rho = P/(R*T)
 *
 */

class density : public compiled_plugin, private compiled_plugin_base
{
   public:
	density();

	inline virtual ~density()
	{
	}
	density(const density& other) = delete;
	density& operator=(const density& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::density";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kCompiled;
	}

   private:
	virtual void Calculate(std::shared_ptr<info<double>> theTargetInfo, unsigned short theThreadIndex);
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<density>(new density());
}
}  // namespace plugin
}  // namespace himan

#endif /* DENSITY_H */
