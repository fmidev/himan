/*
 * plugin_container.cpp
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 */

#include "plugin_container.h"
#include <stdexcept>
#include <dlfcn.h>

using namespace hilpee;

plugin_container::plugin_container(void* theLibraryHandle, std::shared_ptr<plugin::hilpee_plugin> thePlugin)
	: itsPlugin(thePlugin),
	  itsLibraryHandle(theLibraryHandle)
{
}

std::shared_ptr<plugin::hilpee_plugin> plugin_container::Plugin()
{
	return itsPlugin;
}

std::shared_ptr<plugin::hilpee_plugin> plugin_container::Clone()
{
	::dlerror(); // clear error handle

	plugin::create_t* create_plugin = (plugin::create_t*) dlsym(itsLibraryHandle, "create");

	if (!create_plugin)
	{
		// itsLogger->Error("Unable to load symbol: " + std::string(dlerror()));
		return 0;
	}

	return create_plugin();

}

void* plugin_container::Library()
{
	return itsLibraryHandle;
}

plugin_container::~plugin_container()
{

	// Close so

	if (dlclose(itsLibraryHandle) != 0)
	{
		throw std::runtime_error("Library close failed");
	}

}
