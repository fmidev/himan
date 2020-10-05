#ifndef TROPOPAUSE_H
#define TROPOPAUSE_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class tropopause
 *
 * @brief calculate cloud parameters for aviation weather going through the following steps
 * The boundary between the troposphere and the stratosphere, where an abrupt change in lapse rate usually occurs, is
 * defined as the lowest level at which the lapse rate decreases to 2 °C/km or less, provided that the average lapse
 * rate between this level and all higher levels within 2 km does not exceed 2 °C/km.
 */

class tropopause : public compiled_plugin, private compiled_plugin_base
{
   public:
	tropopause();

	inline virtual ~tropopause()
	{
	}
	tropopause(const tropopause& other) = delete;
	tropopause& operator=(const tropopause& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::tropopause";
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
	return std::make_shared<tropopause>();
}
}  // namespace plugin
}  // namespace himan

#endif /* TROPOPAUSE_PLUGIN_H */
