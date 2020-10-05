#ifndef HIMAN_TAF_H
#define HIMAN_TAF_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class himan_taf
 *
 * @brief calculate cloud parameters for aviation weather
 *
 */

class himan_taf : public compiled_plugin, private compiled_plugin_base
{
   public:
	himan_taf();

	inline virtual ~himan_taf()
	{
	}
	himan_taf(const himan_taf& other) = delete;
	himan_taf& operator=(const himan_taf& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::himan_taf";
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
	return std::make_shared<himan_taf>();
}
}  // namespace plugin
}  // namespace himan

#endif /* HIMAN_TAF_PLUGIN_H */
