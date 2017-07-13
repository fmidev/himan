/*
 * cloud_code.h
 *
 */

#ifndef CLOUD_CODE_H
#define CLOUD_CODE_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
class cloud_code : public compiled_plugin, private compiled_plugin_base
{
   public:
	cloud_code();

	inline virtual ~cloud_code() {}
	cloud_code(const cloud_code& other) = delete;
	cloud_code& operator=(const cloud_code& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const { return "himan::plugin::cloud_code"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(1, 0); }
   private:
	virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::shared_ptr<cloud_code>(new cloud_code()); }
}  // namespace plugin
}  // namespace himan

#endif /* CLOUD_CODE_H */
