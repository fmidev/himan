/**
 * @file luatool.h
 *
 *
 */

#ifndef LUATOOL_H
#define LUATOOL_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"
#include "himan_common.h"

extern "C"
{
#include <lua.h>
}

namespace luabind
{
namespace detail
{
namespace has_get_pointer_
{
template <class T>
T* get_pointer(std::shared_ptr<T> const& p)
{
	return p.get();
}
}  // namespace has_get_pointer_
}  // namespace detail
}  // namespace luabind

#include <luabind/object.hpp>

namespace himan
{
namespace plugin
{
class luatool : public compiled_plugin, public compiled_plugin_base
{
   public:
	luatool();
	virtual ~luatool();

	virtual std::string ClassName() const override
	{
		return "himan::plugin::luatool";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kCompiled;
	}

	void Process(std::shared_ptr<const plugin_configuration> configuration);

	std::shared_ptr<info<double>> FetchInfo(const forecast_time& theTime, const level& theLevel,
	                                        const param& theParam) const;
	std::shared_ptr<info<double>> FetchInfo(const forecast_time& theTime, const level& theLevel, const param& theParam,
	                                        const forecast_type& theType) const;
	std::shared_ptr<info<double>> FetchInfoWithArgs(const luabind::object& o) const;

	luabind::object Fetch(const forecast_time& theTime, const level& theLevel, const param& theParam) const;
	luabind::object Fetch(const forecast_time& theTime, const level& theLevel, const param& theParam,
	                      const forecast_type& theType = forecast_type(kDeterministic)) const;
	luabind::object Fetch(const forecast_time& theTime, const level& theLevel, const param& theParam,
	                      const forecast_type& theType, const producer& prod, const std::string& geomName) const;
	luabind::object FetchWithArgs(const luabind::object& o) const;

	void WriteToFile(const std::shared_ptr<info<double>> targetInfo, write_options opts = write_options()) override;
	void WriteToFile(const std::shared_ptr<info<double>> targetInfo);

   private:
	void Calculate(std::shared_ptr<info<double>> theTargetInfo, unsigned short theThreadIndex) override;
	void InitLua();
	void ResetVariables(std::shared_ptr<info<double>> myTargetInfo);
	bool ReadFile(const std::string& luaFile);

	write_options itsWriteOptions;
};

extern "C" std::shared_ptr<luatool> create()
{
	return std::make_shared<luatool>();
}
}  // namespace plugin
}  // namespace himan

#endif /* LUATOOL_H */
