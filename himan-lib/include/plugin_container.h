/*
 * plugin_container.h
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 */

#ifndef PLUGIN_CONTAINER_H
#define PLUGIN_CONTAINER_H

#include <string>
#include "himan_common.h"
#include "himan_plugin.h"

namespace himan
{

class plugin_container
{
public:
    plugin_container() {}

    plugin_container(void* theLibraryHandle, std::shared_ptr<plugin::himan_plugin> thePlugin);

    ~plugin_container();

    std::shared_ptr<plugin::himan_plugin> Plugin();
    std::shared_ptr<plugin::himan_plugin> Clone();

    void* Library();

private:
    std::shared_ptr<plugin::himan_plugin> itsPlugin;
    void* itsLibraryHandle;

};

} // namespace himan

#endif /* PLUGIN_CONTAINER_H */
