/*
 * plugin_factory.cpp
 *
 */

#include "plugin_factory.h"
#include "logger.h"
#include "util.h"
#include <cstdlib>
#include <dlfcn.h>

#define BOOST_FILESYSTEM_NO_DEPRECATED

#include <boost/filesystem.hpp>

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
		std::vector<std::string> paths = util::Split(std::string(path), ":", false);

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

std::vector<std::shared_ptr<himan_plugin>> plugin_factory::Plugins(HPPluginClass pluginClass)
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
	// Try to find the requested plugin twice. Populate the plugin registry if the plugin is not found.
	for (int i = 0; i < 2; i++)
	{
		for (size_t i = 0; i < itsPluginFactory.size(); i++)
		{
			if ((itsPluginFactory[i]->Plugin()->ClassName() == theClassName) ||
			    (itsPluginFactory[i]->Plugin()->ClassName() == "himan::plugin::" + theClassName))
			{
				return itsPluginFactory[i]->Clone();
			}
		}
		ReadPlugins(theClassName);
	}
	throw std::runtime_error("plugin_factory: Unknown plugin clone operation requested: " + theClassName);
}

/*
 * ReadPlugins()
 *
 * Read plugins from defined paths. Will try to load all files in given directories
 * that end with .so. Will not ascend to child directories (equals to "--max-depth 1").
 */

void plugin_factory::ReadPlugins(const std::string& pluginName)
{
	std::lock_guard<std::mutex> lock(itsPluginMutex);

	using namespace boost::filesystem;

	directory_iterator end_iter;

	for (size_t i = 0; i < itsPluginSearchPath.size(); i++)
	{
		path p(itsPluginSearchPath[i]);

		try
		{
			if (exists(p) && is_directory(p))
			{
				for (directory_iterator dir_iter(p); dir_iter != end_iter; ++dir_iter)
				{
					if (dir_iter->path().filename().extension().string() == ".so" &&
					    ((dir_iter->path().stem().string() == "lib" + pluginName) || pluginName.empty()))
					{
						Load(dir_iter->path().string());

						if (!pluginName.empty()) return;
					}
				}
			}
		}

		catch (const filesystem_error& ex)
		{
			itsLogger.Error(ex.what());
		}
	}
}

bool plugin_factory::Load(const std::string& thePluginFileName)
{
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

	auto p = create_plugin();

	for (size_t i = 0; i < itsPluginFactory.size(); i++)
	{
		if (p->ClassName() == itsPluginFactory[i]->Plugin()->ClassName())
		{
			itsLogger.Trace("Plugin '" + p->ClassName() + "' found more than once, skipping one found from '" +
			                thePluginFileName + "'");
			dlclose(theLibraryHandle);

			return true;
		}
	}

	std::shared_ptr<plugin_container> mc = std::shared_ptr<plugin_container>(new plugin_container(theLibraryHandle, p));

	itsPluginFactory.push_back(mc);

	itsLogger.Trace("Load " + thePluginFileName);

	return true;
}
