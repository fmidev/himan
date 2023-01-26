/*
 * plugin_factory.cpp
 *
 */

#include "plugin_factory.h"
#include "logger.h"
#include "util.h"
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>

using namespace himan;
using namespace himan::plugin;

std::unique_ptr<plugin_factory> plugin_factory::itsInstance;

plugin_factory* plugin_factory::Instance()
{
	if (!itsInstance)
	{
		itsInstance = std::unique_ptr<plugin_factory>(new plugin_factory());
	}

	return itsInstance.get();
}

plugin_factory::plugin_factory() : itsPluginSearchPath(), itsLogger(logger("plugin_factory"))
{
	const char* path = std::getenv("HIMAN_LIBRARY_PATH");

	if (path != NULL)
	{
		std::vector<std::string> paths = util::Split(std::string(path), ":");

		itsPluginSearchPath.insert(itsPluginSearchPath.end(), paths.begin(), paths.end());
	}
	else
	{
		itsLogger.Trace(
		    "Environment variable HIMAN_LIBRARY_PATH not set -- search plugins only from pre-defined locations");
	}

	itsPluginSearchPath.push_back(".");
	itsPluginSearchPath.push_back("/usr/lib64/himan-plugins");  // Default RPM location
}

std::vector<std::shared_ptr<himan_plugin>> plugin_factory::Plugins()
{
	ReadPlugins();

	std::vector<std::shared_ptr<himan_plugin>> thePlugins(itsPluginFactory.size());

	for (size_t i = 0; i < itsPluginFactory.size(); i++)
	{
		thePlugins[i] = itsPluginFactory[i]->Plugin();
	}

	return thePlugins;
}

std::shared_ptr<himan_plugin> plugin_factory::Plugin(const std::string& theClassName)
{
	std::lock_guard<std::mutex> lock(itsPluginMutex);
	// Try to find the requested plugin twice. Populate the plugin registry if the plugin is not found.
	for (int i = 0; i < 2; i++)
	{
		for (size_t j = 0; j < itsPluginFactory.size(); j++)
		{
			if ((itsPluginFactory[j]->Plugin()->ClassName() == theClassName) ||
			    (itsPluginFactory[j]->Plugin()->ClassName() == "himan::plugin::" + theClassName))
			{
				return itsPluginFactory[j]->Clone();
			}
		}
		ReadPlugins(theClassName);
	}
	itsLogger.Error("Unable to find plugin '" + theClassName + "'");
	throw kPluginNotFound;
}

/*
 * ReadPlugins()
 *
 * Read plugins from defined paths. Will try to load all files in given directories
 * that end with .so. Will not ascend to child directories (equals to "--max-depth 1").
 */

void plugin_factory::ReadPlugins(const std::string& pluginName)
{
	namespace fs = std::filesystem;

	for (const auto& pluginPath : itsPluginSearchPath)
	{
		fs::path p(pluginPath);

		try
		{
			if (fs::exists(p) && fs::is_directory(p))
			{
				for (const auto& entry : fs::directory_iterator(p))
				{
					if (entry.path().extension().string() == ".so" &&
					    (entry.path().stem().string() == "lib" + pluginName || pluginName.empty()))
					{
						Load(entry.path().string());

						if (!pluginName.empty())
						{
							return;
						}
					}
				}
			}
		}
		catch (const fs::filesystem_error& ex)
		{
			itsLogger.Error(ex.what());
		}
	}
}

bool plugin_factory::Load(const std::string& thePluginFileName)
{
	/*
	 * Check if library has been loaded already. We cannot first load the library and
	 * then call its ClassName() because in some occasions that causes crashes when
	 * API/ABI is not compatible.
	 *
	 * Note: this assumes that library filename matches ClassName as such:
	 * filename: libNAME.so
	 * ClassName: himan::plugin::NAME
	 */

	const std::string stem = std::filesystem::path(thePluginFileName).stem().string().substr(3, std::string::npos);

	if (std::find_if(itsPluginFactory.begin(), itsPluginFactory.end(),
	                 [&stem](std::shared_ptr<plugin_container>& cont)
	                 { return cont->Plugin()->ClassName() == "himan::plugin::" + stem; }) != itsPluginFactory.end())
	{
		itsLogger.Debug("Plugin '" + stem + "' found more than once, skipping one found from '" + thePluginFileName +
		                "'");
		return true;
	}

	/*
	 * Open libraries with
	 *
	 *   RTLD_LAZY
	 * We don't specify the ordering of the plugin load -- usually it is alphabetical
	 * but not necessarily so. With RTLD_LAZY the symbols aren't checked during load
	 * which means that the first loaded plugin can refer to functions defined in the
	 * last loaded plugin without compiler issuing warnings like
	 *  <file>.so undefined symbol: ...
	 *
	 *   RTLD_GLOBAL
	 * We need this because core library (himan-lib) need to access aux plugins
	 * and the plugin symbol information needs to be propagated throughout the
	 * plugins system (or using aux plugins in core lib will fail).
	 */

	void* theLibraryHandle = dlopen(thePluginFileName.c_str(), RTLD_LAZY | RTLD_GLOBAL);

	if (!theLibraryHandle)
	{
		itsLogger.Error("Unable to load plugin '" + thePluginFileName + "': " + std::string(dlerror()));
		return false;
	}

	dlerror();  // clear error handle

	create_t* create_plugin = reinterpret_cast<create_t*>(dlsym(theLibraryHandle, "create"));

	if (!create_plugin)
	{
		itsLogger.Error("Unable to load symbol: " + std::string(dlerror()));
		return false;
	}

	itsPluginFactory.push_back(std::make_shared<plugin_container>(theLibraryHandle, create_plugin()));

	itsLogger.Trace("Load " + thePluginFileName);

	return true;
}
