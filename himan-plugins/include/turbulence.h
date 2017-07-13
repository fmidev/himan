/**
 * @file turbulence.h
 *
 */

#ifndef TURBULENCE_H
#define TURBULENCE_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class turbulence
 *
 * @brief Calculate the turbulence index
 *
 */

class turbulence : public compiled_plugin, private compiled_plugin_base
{
   public:
	turbulence();

	inline virtual ~turbulence() {}
	turbulence(const turbulence& other) = delete;
	turbulence& operator=(const turbulence& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const { return "himan::plugin::turbulence"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(0, 1); }
   private:
	virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::make_shared<turbulence>(); }
}  // namespace plugin
}  // namespace himan

#endif /* EXAMPLE_PLUGIN_H */
