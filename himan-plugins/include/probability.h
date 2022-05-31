#ifndef PROBABILITY_PLUGIN_H
#define PROBABILITY_PLUGIN_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"
#include "probability_core.h"

namespace himan
{
namespace plugin
{
class probability : public compiled_plugin, private compiled_plugin_base
{
   public:
	probability();

	virtual ~probability();

	probability(const probability& other) = delete;
	probability& operator=(const probability& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::probability";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kCompiled;
	}

   private:
	virtual void Calculate(std::shared_ptr<info<float>> theTargetInfo, unsigned short theThreadIndex);
	void Worker(std::shared_ptr<info<float>> myTargetInfo, short threadIndex);
	PROB::partial_param_configuration GetTarget();

	std::vector<PROB::partial_param_configuration> itsParamConfigurations;
};

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<probability>();
}
}  // namespace plugin
}  // namespace himan

// PROBABILITY_PLUGIN_H
#endif
