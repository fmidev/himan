/*
 * fog.h
 *
 */

#ifndef FOG_H
#define FOG_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
class fog : public compiled_plugin, private compiled_plugin_base
{
   public:
	fog();

	inline virtual ~fog()
	{
	}
	fog(const fog& other) = delete;
	fog& operator=(const fog& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::fog";
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
	return std::shared_ptr<fog>(new fog());
}
}  // namespace plugin
}  // namespace himan

#endif /* FOG_H */
