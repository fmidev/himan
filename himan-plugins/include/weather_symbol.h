/*
 * @file weather_symbol.h
 *
 */

#ifndef WEATHER_SYMBOL_H
#define WEATHER_SYMBOL_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"
#include <boost/bimap.hpp>

namespace himan
{
namespace plugin
{
/**
 * @class weather_symbol
 *
 * @brief Calculate hessaa
 *
 */

class weather_symbol : public compiled_plugin, private compiled_plugin_base
{
   public:
	weather_symbol();

	inline virtual ~weather_symbol() {}
	weather_symbol(const weather_symbol& other) = delete;
	weather_symbol& operator=(const weather_symbol& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const { return "himan::plugin::weather_symbol"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(1, 0); }
   private:
	virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);
	double rain_form(double rr);
	double rain_type(double rr);
	double weather_type(double rr);
	double cloud_type(double cloud);

	std::map<double, double> cloudMap;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::shared_ptr<weather_symbol>(new weather_symbol()); }
}  // namespace plugin
}  // namespace himan

#endif /* weather_symbol */
