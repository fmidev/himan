#ifndef AUTO_TAF_H
#define AUTO_TAF_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{

/**
 * @class auto_taf
 *
 * @brief calculate cloud parameters for aviation weather
 *
 */

class auto_taf : public compiled_plugin, private compiled_plugin_base
{
   public:
	auto_taf();

	inline virtual ~auto_taf() {}
	auto_taf(const auto_taf& other) = delete;
	auto_taf& operator=(const auto_taf& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const { return "himan::plugin::pot"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(0, 1); }
   private:
	virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);
	bool itsStrictMode;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::make_shared<pot>(); }
}  // namespace plugin
}  // namespace himan

#endif /* AUTO_TAF_PLUGIN_H */
