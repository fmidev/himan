/*
 * monin_obukhov.h
 *
 */

#ifndef MONIN_OBUKHOV_H
#define MONIN_OBUKHOV_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class monin_obukhov
 *
 * @brief Calculate the inverse of the monin-obukhov length.
 *
 */

class monin_obukhov : public compiled_plugin, private compiled_plugin_base
{
   public:
	monin_obukhov();

	inline virtual ~monin_obukhov() {}
	monin_obukhov(const monin_obukhov& other) = delete;
	monin_obukhov& operator=(const monin_obukhov& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const { return "himan::plugin::monin_obukhov"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(0, 1); }
   private:
	virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::shared_ptr<monin_obukhov>(new monin_obukhov()); }
}  // namespace plugin
}  // namespace himan

#endif /* MONIN_OBUKHOV_H */
