/**
 * @file windvector.h
 *
 */

#ifndef WINDVECTOR_H
#define WINDVECTOR_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

#include "windvector.cuh"  // need to have this here because of HPTargetType

namespace himan
{
namespace plugin
{
class windvector : public compiled_plugin, private compiled_plugin_base
{
   public:
	windvector();

	virtual ~windvector() = default;

	windvector(const windvector& other) = delete;
	windvector& operator=(const windvector& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::windvector";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kCompiled;
	}

   private:
	std::shared_ptr<info<float>> FetchOne(const forecast_time& theTime, const level& theLevel, const param& theParam,
	                                      const forecast_type& theType = forecast_type(kDeterministic),
	                                      bool returnPacked = false) const;

	virtual void Calculate(std::shared_ptr<info<float>> theTargetInfo, unsigned short theThreadIndex);

	HPWindVectorTargetType itsCalculationTarget;
	bool itsVectorCalculation;
	bool itsReverseCalculation;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<windvector>();
}
}  // namespace plugin
}  // namespace himan

#endif /* WINDVECTOR */
