#pragma once

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class stability_simple
 *
 * @brief Calculate k-index and other indexes that describe that stability_simple of the atmosphere.
 *
 */

class stability_simple : public compiled_plugin, private compiled_plugin_base
{
   public:
	stability_simple();

	inline virtual ~stability_simple()
	{
	}
	stability_simple(const stability_simple& other) = delete;
	stability_simple& operator=(const stability_simple& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::stability_simple";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kCompiled;
	}

   private:
	void Calculate(std::shared_ptr<info<double>> theTargetInfo, unsigned short theThreadIndex);
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<stability_simple>(new stability_simple());
}
}  // namespace plugin
}  // namespace himan
