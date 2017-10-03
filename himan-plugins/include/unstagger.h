/*
 * unstagger.h
 *
 */

#ifndef UNSTAGGER_H
#define UNSTAGGER_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class unstagger
 *
 * @brief Calculate the co-located velocity field for U and V
 *
 */

class unstagger : public compiled_plugin, private compiled_plugin_base
{
   public:
	unstagger();

	inline virtual ~unstagger() {}
	unstagger(const unstagger& other) = delete;
	unstagger& operator=(const unstagger& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const { return "himan::plugin::unstagger"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(0, 1); }
   private:
	virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::shared_ptr<unstagger>(new unstagger()); }
}  // namespace plugin
}  // namespace himan

#endif /* UNSTAGGER_H */
