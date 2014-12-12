/**
 * @file luatool.h
 *
 * @date 2014-12-01
 * @author partio
 *
*/

#ifndef LUATOOL_H
#define LUATOOL_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"
#include "himan_common.h"

extern "C"
{
#include <lua5.2/lua.h>
}

namespace himan
{
namespace plugin
{

class luatool : public compiled_plugin, public compiled_plugin_base
{
public:
	luatool();
	virtual ~luatool();

	virtual std::string ClassName() const
	{
		return "himan::plugin::luatool";
	}

	virtual HPPluginClass PluginClass() const
	{
		return kCompiled;
	}

	virtual HPVersionNumber Version() const
	{
		return HPVersionNumber(0, 1);
	}

	void Process(std::shared_ptr<const plugin_configuration> configuration);

	std::shared_ptr<info> Fetch(const forecast_time& theTime, const level& theLevel, const param& theParam) const;
	
protected:
	/* These functions exists because we need to stop himan
	 * from writing data to disk when calculation finished.
	 *
	 * All data write in luatool should be initiated from the
	 * lua scripts!
	 */

	virtual void Finish() const;
	virtual void Run(info_t myTargetInfo, unsigned short threadIndex);

private:
    virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);

	void InitLua(std::shared_ptr<info> myTargetInfo);
	bool ReadFile(const std::string& luaFile);

	lua_State* L;	
};

#ifndef HIMAN_AUXILIARY_INCLUDE

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<luatool> ();
}

#endif /* HIMAN_AUXILIARY_INCLUDE */

} // namespace plugin
} // namespace himan

#endif /* LUATOOL_H */