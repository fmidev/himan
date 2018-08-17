/**
 * @file windvector.h
 *
 */

#ifndef WINDVECTOR_H
#define WINDVECTOR_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

#include "windvector.cuh"  // need to have this here because of HPTargetType

class NFmiArea;

namespace himan
{
namespace plugin
{
class windvector : public compiled_plugin, private compiled_plugin_base
{
   public:
	windvector();

	inline virtual ~windvector()
	{
	}
	windvector(const windvector& other) = delete;
	windvector& operator=(const windvector& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const
	{
		return "himan::plugin::windvector";
	}
	virtual HPPluginClass PluginClass() const
	{
		return kCompiled;
	}
	virtual HPVersionNumber Version() const
	{
		return HPVersionNumber(1, 1);
	}

   protected:
	virtual std::shared_ptr<info> Fetch(const forecast_time& theTime, const level& theLevel, const param& theParam,
	                                    const forecast_type& theType = forecast_type(kDeterministic),
	                                    bool returnPacked = false) const override;

   private:
	virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);

	HPWindVectorTargetType itsCalculationTarget;
	bool itsVectorCalculation;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<windvector>(new windvector());
}
}  // namespace plugin
}  // namespace himan

#endif /* WINDVECTOR */
