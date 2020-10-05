/**
 * @file vvms.h
 *
 */

#ifndef VVMS_H
#define VVMS_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class vvms
 *
 * @brief Calculate vertical velocity in m/s.
 *
 */

class vvms : public compiled_plugin, private compiled_plugin_base
{
   public:
	vvms();
	virtual ~vvms() = default;

	vvms(const vvms& other) = delete;
	vvms& operator=(const vvms& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::vvms";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kCompiled;
	}

   private:
	virtual void Calculate(std::shared_ptr<info<float>> theTargetInfo, unsigned short theThreadIndex);
	float itsScale;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<vvms>();
}
}  // namespace plugin
}  // namespace himan

#endif /* VVMS_H */
