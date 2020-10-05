/*
 * ncl.h
 *
 */

#ifndef NCL_H
#define NCL_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class ncl
 *
 * @brief Calculate ...
 *
 */

class ncl : public compiled_plugin, private compiled_plugin_base
{
   public:
	ncl();

	inline virtual ~ncl()
	{
	}
	ncl(const ncl& other) = delete;
	ncl& operator=(const ncl& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::ncl";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kCompiled;
	}

   private:
	virtual void Calculate(std::shared_ptr<info<double>> theTargetInfo, unsigned short theThreadIndex);
	int itsTargetTemperature;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<ncl>(new ncl());
}
}  // namespace plugin
}  // namespace himan

#endif /* NCL_H */
