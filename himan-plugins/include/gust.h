/*
 * gust.h
 *
 */

#ifndef GUST_PLUGIN_H
#define GUST_PLUGIN_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class gust
 *
 * @brief Calculate gust wind speed.
 *
 */

class gust : public compiled_plugin, private compiled_plugin_base
{
   public:
	gust();

	inline virtual ~gust()
	{
	}
	gust(const gust& other) = delete;
	gust& operator=(const gust& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::gust";
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
	return std::shared_ptr<gust>(new gust());
}
}  // namespace plugin
}  // namespace himan

#endif /* GUST_PLUGIN_H */
