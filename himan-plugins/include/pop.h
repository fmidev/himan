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

	inline virtual ~pop() {}
	pop(const pop& other) = delete;
	pop& operator=(const pop& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const { return "himan::plugin::pop"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(0, 1); }
   private:
	virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);

	std::string itsECEPSGeom;
	std::string itsECGeom;
	std::string itsPEPSGeom;
	std::string itsHirlamGeom;
	std::string itsHarmonieGeom;
	std::string itsGFSGeom;
};

extern "C" std::shared_ptr<himan_plugin> create() { return std::make_shared<pop>(); }
}  // namespace plugin
}  // namespace himan

#endif /* POP_H */
