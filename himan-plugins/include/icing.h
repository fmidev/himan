/*
 * @file icing.h
 *
 */

#ifndef ICING_H
#define ICING_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class icing
 *
 * @brief Calculate icing index for hybrid levels.
 *
 */

class icing : public compiled_plugin, private compiled_plugin_base
{
   public:
	icing();

	inline virtual ~icing() = default;
	icing(const icing& other) = delete;
	icing& operator=(const icing& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::icing";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kCompiled;
	}

   private:
	virtual void Calculate(std::shared_ptr<info<float>> theTargetInfo, unsigned short theThreadIndex) override;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<icing>();
}
}  // namespace plugin
}  // namespace himan

#endif /* ICING */
