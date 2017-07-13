/*
 * weather_code_1.h
 *
 */

#ifndef WEATHER_CODE_1_H
#define WEATHER_CODE_1_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class weather_code_1
 *
 * @brief Calculate ...
 *
 */

class weather_code_1 : public compiled_plugin, private compiled_plugin_base
{
   public:
	weather_code_1();

	inline virtual ~weather_code_1() {}
	weather_code_1(const weather_code_1& other) = delete;
	weather_code_1& operator=(const weather_code_1& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const { return "himan::plugin::weather_code_1"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(1, 0); }
   private:
	virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);
	std::shared_ptr<info> FetchSourceRR(const forecast_time& wantedTime, const level& wantedLevel);
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::shared_ptr<weather_code_1>(new weather_code_1()); }
}  // namespace plugin
}  // namespace himan

#endif /* weather_code_1_H */
