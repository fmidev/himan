/**
 * @file tke.h
 *
 */

#ifndef POT_H
#define POT_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class TKE
 *
 * @brief Calculate Turbulent Kinetic Energy (TKE) in the Atmospheric Boundary Layer
 *
 */

class tke : public compiled_plugin, private compiled_plugin_base
{
   public:
	tke();

	inline virtual ~tke()
	{
	}
	tke(const tke& other) = delete;
	tke& operator=(const tke& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::tke";
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
	return std::make_shared<tke>();
}
}  // namespace plugin
}  // namespace himan

#endif /* POT_PLUGIN_H */
