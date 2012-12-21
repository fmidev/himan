/*
 * plugin_factory.cpp
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 */

#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <dlfcn.h>

using namespace hilpee;
using namespace hilpee::plugin;

plugin_factory* plugin_factory::itsInstance = NULL;

plugin_factory* plugin_factory::Instance()
{
	if (!itsInstance)
	{
		itsInstance = new plugin_factory();
	}

	return itsInstance;
}


plugin_factory::plugin_factory() : itsPluginSearchPath()
{

	itsLogger = logger_factory::Instance()->GetLog("plugin_factory");

	itsPluginSearchPath.push_back("../hilpee-plugins/lib");

	ReadPlugins();
}

// Hide constructor
plugin_factory::~plugin_factory() {}

std::vector<std::shared_ptr<hilpee_plugin>> plugin_factory::Plugins(HPPluginClass pluginClass)
{

	std::vector<std::shared_ptr<hilpee_plugin>> thePlugins;

for (auto theContainer : itsPluginFactory)
	{
		if (pluginClass == theContainer->Plugin()->PluginClass())
		{
			thePlugins.push_back(theContainer->Plugin());
		}

		else if (pluginClass == kUnknownPlugin)
		{
			thePlugins.push_back(theContainer->Plugin());
		}
	}

	return thePlugins;
}

std::vector<std::shared_ptr<hilpee_plugin>> plugin_factory::CompiledPlugins()
{
	return Plugins(kCompiled);
}
std::vector<std::shared_ptr<hilpee_plugin>> plugin_factory::AuxiliaryPlugins()
{
	return Plugins(kAuxiliary);
}
std::vector<std::shared_ptr<hilpee_plugin>> plugin_factory::InterpretedPlugins()
{
	return Plugins(kInterpreted);
}

/*
 * Plugin()
 *
 * Return instance of the requested plugin if found. Caller must cast
 * the plugin to the derived class. If second argument is true, a new
 * instance is created and returned. Otherwise function behaves like
 * a regular factory pattern and return one known instance to each
 * caller (this is suitable only in non-threaded functions).
 */

std::shared_ptr<hilpee_plugin> plugin_factory::Plugin(const std::string& theClassName, bool theNewInstance)
{

for (auto theContainer : itsPluginFactory)
	{

		if ((theContainer->Plugin()->ClassName() == theClassName) ||
		        (theContainer->Plugin()->ClassName() == "hilpee::plugin::" + theClassName))
		{
			if (theNewInstance)
			{
				return theContainer->Clone();
			}
			else
			{
				return theContainer->Plugin();
			}
		}
	}

	return 0;
}

/*
 * ReadPlugins()
 *
 * Read plugins from defined paths. Will try to load all files in given directories
 * that end with .so. Will not ascend to child directories (equals to "--max-depth 1").
 */

void plugin_factory::ReadPlugins()
{

	using namespace boost::filesystem;

	directory_iterator end_iter;

for (std::string thePath : itsPluginSearchPath)
	{
		path p (thePath);

		try
		{
			if (exists(p) && is_directory(p))      // is p a directory?
			{

				for ( directory_iterator dir_iter(p) ; dir_iter != end_iter ; ++dir_iter)
				{
					if (dir_iter->path().filename().extension().string() == ".so")
					{
						Load(dir_iter->path().string());
					}
				}
			}
		}

		catch (const filesystem_error& ex)
		{
			itsLogger->Error(ex.what());
		}
	}

}

bool plugin_factory::Load(const std::string& thePluginFileName)
{

	itsLogger->Debug("Load " + thePluginFileName);

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
	 * We need this because core library (hilpee-lib) need to access aux plugins
	 * and the plugin symbol information needs to be propagated throughout the
	 * plugins system (or using aux plugins in core lib will fail).
	 */

	void* theLibraryHandle = dlopen(thePluginFileName.c_str(), RTLD_LAZY | RTLD_GLOBAL);

	if (!theLibraryHandle)
	{
		itsLogger->Error("Unable to load plugin: " + std::string(dlerror()));
		return false;
	}

	dlerror(); // clear error handle

	create_t* create_plugin = (create_t*) dlsym(theLibraryHandle, "create");

	if (!create_plugin)
	{
		itsLogger->Error("Unable to load symbol: " + std::string(dlerror()));
		return false;
	}

	std::shared_ptr<plugin_container> mc = std::shared_ptr<plugin_container> (new plugin_container(theLibraryHandle, create_plugin()));

	itsPluginFactory.push_back(mc);

	return true;
}
