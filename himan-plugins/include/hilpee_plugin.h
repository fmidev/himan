/*
 * hilpee_plugin.h
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 *
 * Top level class for all plugins. (Note: not for core-lib).
 */

#ifndef HILPEE_PLUGIN_H
#define HILPEE_PLUGIN_H

#include <string>
#include "hilpee_common.h"
#include "logger.h"

#ifdef HAVE_CPP11
#include <memory> // for std::shared_ptr (g++ 4.6)
#else
#include <tr1/memory> // for std::shared_ptr
#endif

namespace hilpee
{

#ifndef HAVE_CPP11
using std::tr1::shared_ptr;
#else
using std::shared_ptr;
#endif

namespace plugin
{

class hilpee_plugin
{
	public:

		inline hilpee_plugin() {};

		inline virtual ~hilpee_plugin() {};

		virtual std::string ClassName() const = 0;

		virtual HPPluginClass PluginClass() const = 0;

		virtual HPVersionNumber Version() const = 0;

	protected:
#ifdef HAVE_CPP11
		std::unique_ptr<logger> itsLogger;
#else
		std::auto_ptr<logger> itsLogger;
#endif
};

// the type of the class factory

typedef shared_ptr<hilpee_plugin> create_t();

}
}

#endif /* HILPEE_PLUGIN_H */
