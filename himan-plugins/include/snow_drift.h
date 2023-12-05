#ifndef SNOW_DRIFT_H
#define SNOW_DRIFT_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
class snow_drift : public compiled_plugin, private compiled_plugin_base
{
   public:
	snow_drift();

	inline virtual ~snow_drift() = default;

	snow_drift(const snow_drift& other) = delete;
	snow_drift& operator=(const snow_drift& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::snow_drift";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kCompiled;
	}

   private:
	void Calculate(std::shared_ptr<info<double>> theTargetInfo, unsigned short theThreadIndex) override;
	bool itsResetSnowDrift;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<snow_drift>();
}
}  // namespace plugin
}  // namespace himan

#endif /* SNOW_DRIFT_H */
