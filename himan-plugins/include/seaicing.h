/*
 * @file seaicing.h
 *
 */

#ifndef SEAICING_H
#define SEAICING_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class seaicing
 *
 * @brief Calculate sea spray icing index.
 *
 */

class seaicing : public compiled_plugin, private compiled_plugin_base
{
   public:
	seaicing();

	inline virtual ~seaicing() {}
	seaicing(const seaicing& other) = delete;
	seaicing& operator=(const seaicing& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const { return "himan::plugin::seaicing"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(0, 1); }
   private:
	virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);
	bool global;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::shared_ptr<seaicing>(new seaicing()); }
}  // namespace plugin
}  // namespace himan

#endif /* SEAICING */
