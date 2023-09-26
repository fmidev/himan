/**
 * @file pop.h
 *
 */

#ifndef POP_H
#define POP_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
class pop : public compiled_plugin, private compiled_plugin_base
{
   public:
	pop();

	inline virtual ~pop()
	{
	}
	pop(const pop& other) = delete;
	pop& operator=(const pop& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::pop";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kCompiled;
	}

   private:
	virtual void Calculate(std::shared_ptr<info<double>> theTargetInfo, unsigned short theThreadIndex);
	std::shared_ptr<himan::info<double>> GetShortProbabilityData(const forecast_time& forecastTime, const level& level,
	                                                             logger& logr);

	std::string itsECEPSGeom;
	std::string itsMEPSGeom;

	bool itsUseMEPS;
};

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<pop>();
}
}  // namespace plugin
}  // namespace himan

#endif /* POP_H */
