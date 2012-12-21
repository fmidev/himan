/*
 * auxiliary_plugin.h
 *
 *  Created on: Nov 19, 2012
 *      Author: partio
 *
 * Auxiliary plugins are used to help the calculation of parameters.
 * They do not calculate parameters themselves.
 *
 * Aux plugins are generally not called from the main executable but
 * from other plugins. In the executable we only reveal the auxiliary
 * plugin interface (described by this file), which generally does not
 * contain any useful function calls since aux plugins are quite varying
 * in nature.
 *
 * If we'd want to call an individual plugin's functions from the
 * executable, we would have to link the executable against the plugin
 * library which destroys the whole notion and idea of a "plugin system."
 *
 */

#ifndef AUXILIARY_PLUGIN_H
#define AUXILIARY_PLUGIN_H

#include "hilpee_plugin.h"
#include "info.h"

namespace hilpee
{
namespace plugin
{

class auxiliary_plugin : public hilpee_plugin
{

	public:
		auxiliary_plugin() {};

		virtual ~auxiliary_plugin() {};

};

}
}

#endif /* AUXILIARY_PLUGIN_H */
