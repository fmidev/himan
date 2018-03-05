/**
 * @file vvms.h
 *
 */

#ifndef VVMS_H
#define VVMS_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class vvms
 *
 * @brief Calculate vertical velocity in m/s.
 *
 */

class vvms : public compiled_plugin, private compiled_plugin_base
{
   public:
	vvms();

	inline virtual ~vvms()
	{
	}
	vvms(const vvms& other) = delete;
	vvms& operator=(const vvms& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const
	{
		return "himan::plugin::vvms";
	}
	virtual HPPluginClass PluginClass() const
	{
		return kCompiled;
	}
	virtual HPVersionNumber Version() const
	{
		return HPVersionNumber(1, 1);
	}

   private:
	virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);
	double itsScale;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<vvms>(new vvms());
}
}  // namespace plugin
}  // namespace himan

#endif /* VVMS_H */
