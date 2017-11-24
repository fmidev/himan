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

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const
	{
		return "himan::plugin::himan_taf";
	}
	virtual HPPluginClass PluginClass() const
	{
		return kCompiled;
	}
	virtual HPVersionNumber Version() const
	{
		return HPVersionNumber(0, 1);
	}

   private:
	virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);
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
