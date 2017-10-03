/*
 * @file hybrid_pressure.h
 *
 */

#ifndef HYBRID_PRESSURE_H
#define HYBRID_PRESSURE_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class hybrid_pressure
 *
 * @brief Calculate pressure on hybrid level.
 *
 */

class hybrid_pressure : public compiled_plugin, private compiled_plugin_base
{
   public:
	hybrid_pressure();

	inline virtual ~hybrid_pressure() {}
	hybrid_pressure(const hybrid_pressure& other) = delete;
	hybrid_pressure& operator=(const hybrid_pressure& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const { return "himan::plugin::hybrid_pressure"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(0, 1); }
	virtual void WriteToFile(const info& targetInfo, write_options opts = write_options()) override;

   private:
	virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::shared_ptr<hybrid_pressure>(new hybrid_pressure()); }
}  // namespace plugin
}  // namespace himan

#endif /* HYBRID_PRESSURE_H */
