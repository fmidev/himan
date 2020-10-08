/*
 * @file hybrid_pressure.h
 *
 */

#ifndef HYBRID_PRESSURE_H
#define HYBRID_PRESSURE_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class hybrid_pressure
 *
 * @brief Calculate pressure on hybrid level.
 *
 */

class hybrid_pressure : public compiled_plugin, private compiled_plugin_base
{
   public:
	hybrid_pressure();
	inline virtual ~hybrid_pressure() = default;

	hybrid_pressure(const hybrid_pressure& other) = delete;
	hybrid_pressure& operator=(const hybrid_pressure& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::hybrid_pressure";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kCompiled;
	}

   private:
	virtual void Calculate(std::shared_ptr<info<float>> theTargetInfo, unsigned short theThreadIndex) override;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<hybrid_pressure>();
}
}  // namespace plugin
}  // namespace himan

#endif /* HYBRID_PRESSURE_H */
