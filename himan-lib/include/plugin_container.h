/*
 * plugin_container.h
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 */

#ifndef PLUGIN_CONTAINER_H
#define PLUGIN_CONTAINER_H

#include <string>
#include "hilpee_common.h"
#include "hilpee_plugin.h"

namespace hilpee
{

class plugin_container
{
	public:
		plugin_container() {}

		plugin_container(void* theLibraryHandle, std::shared_ptr<plugin::hilpee_plugin> thePlugin);

		~plugin_container();

		std::shared_ptr<plugin::hilpee_plugin> Plugin();
		std::shared_ptr<plugin::hilpee_plugin> Clone();

		void* Library();

	private:
		std::shared_ptr<plugin::hilpee_plugin> itsPlugin;
		void* itsLibraryHandle;

};

} // namespace hilpee

#endif /* PLUGIN_CONTAINER_H */
