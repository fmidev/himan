#ifndef POT_H
#define POT_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class pot
 *
 * @brief Calculate the thunder probability
 *
 */

class pot : public compiled_plugin, private compiled_plugin_base
{
   public:
	pot();

	inline virtual ~pot()
	{
	}
	pot(const pot& other) = delete;
	pot& operator=(const pot& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::pot";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kCompiled;
	}

   private:
	virtual void Calculate(std::shared_ptr<info<double>> theTargetInfo, unsigned short theThreadIndex);
	bool itsStrictMode;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<pot>();
}
}  // namespace plugin
}  // namespace himan

#endif /* POT_PLUGIN_H */
