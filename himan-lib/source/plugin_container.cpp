/*
 * plugin_container.cpp
 *
 */

#include "plugin_container.h"
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>

using namespace himan;

plugin_container::plugin_container() : itsLibraryHandle(0) {}
plugin_container::plugin_container(void* theLibraryHandle, std::shared_ptr<plugin::himan_plugin> thePlugin)
    : itsPlugin(thePlugin), itsLibraryHandle(theLibraryHandle)
{
}

std::shared_ptr<plugin::himan_plugin> plugin_container::Plugin() { return itsPlugin; }
std::shared_ptr<plugin::himan_plugin> plugin_container::Clone()
{
	::dlerror();  // clear error handle

	assert(itsLibraryHandle);
	plugin::create_t* create_plugin = reinterpret_cast<plugin::create_t*>(dlsym(itsLibraryHandle, "create"));

	if (!create_plugin)
	{
		// itsLogger->Error("Unable to load symbol: " + std::string(dlerror()));
		// return 0;
		throw std::runtime_error("himan::plugin_container: Unable to clone plugin");
	}

	return create_plugin();
}

void* plugin_container::Library() { return itsLibraryHandle; }
plugin_container::~plugin_container()
{
	// Close libraries
	// This function is called only when himan exits

	if (dlclose(itsLibraryHandle) != 0)
	{
		std::cerr << "Library close failed" << std::endl;
	}
}
