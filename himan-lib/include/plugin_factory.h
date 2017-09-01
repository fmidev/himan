/**
 * @file plugin_factory.h
 *
 */

#ifndef PLUGIN_FACTORY_H
#define PLUGIN_FACTORY_H

#include "plugin_container.h"
#include <memory>
#include <mutex>
#include <vector>

#define GET_PLUGIN(P) std::dynamic_pointer_cast<himan::plugin::P>(plugin_factory::Instance()->Plugin(#P))

namespace himan
{
class plugin_factory
{
   public:
	~plugin_factory() {}
	static plugin_factory* Instance();

	std::vector<std::shared_ptr<plugin::himan_plugin>> Plugins(HPPluginClass = kUnknownPlugin);  // Kinda ugly

	/**
	 * @brief Return instance of the requested plugin if found. Caller must cast
	 * the plugin to the derived class.
	 */

	std::shared_ptr<plugin::himan_plugin> Plugin(const std::string& theClassName);

   private:
	// Hide constructor
	plugin_factory();

	plugin_factory(const plugin_factory&) = delete;

	static std::unique_ptr<plugin_factory> itsInstance;

	void ReadPlugins(const std::string& pluginName = "");

	bool Load(const std::string& thePluginFileName);

	void Unload();

	std::vector<std::shared_ptr<plugin_container>> itsPluginFactory;

	std::vector<std::string> itsPluginSearchPath;
	logger itsLogger;

	std::mutex itsPluginMutex;
};

}  // namespace himan

#endif /* PLUGIN_FACTORY_H */
