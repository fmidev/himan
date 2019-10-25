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

	virtual std::string ClassName() const
	{
		return "himan::plugin::probability";
	}
	virtual HPPluginClass PluginClass() const
	{
		return kCompiled;
	}

   private:
	virtual void Calculate(std::shared_ptr<info<float>> theTargetInfo, unsigned short theThreadIndex);

	std::vector<PROB::partial_param_configuration> itsParamConfigurations;

	int itsEnsembleSize;
	int itsMaximumMissingForecasts;
	bool itsUseLaggedEnsemble;
	time_duration itsLag;
	time_duration itsLagStep;
};

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<probability>();
}
}  // plugin
}  // himan

// PROBABILITY_PLUGIN_H
#endif
