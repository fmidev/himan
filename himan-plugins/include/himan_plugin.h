/**
 * @file himan_plugin.h
 *
 * @brief Top level class for all plugins. (not for core-lib).
 *
 */

#ifndef HIMAN_PLUGIN_H
#define HIMAN_PLUGIN_H

#include "himan_common.h"
#include "logger.h"

#if defined __clang__

#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"

#endif

/**
 * @namespace himan
 * @brief Bottom-level namespace for all himan-related stuff
 */

namespace himan
{
/**
 * @namespace himan::plugin
 * @brief Namespace for all himan plugins
 */

namespace plugin
{
/**
 * @class himan_plugin Top class interface for all plugins
 *
 * @brief This class acts as an interface, meaning that it declares a bunch of functions
 * but does not give definitions.
 */

class himan_plugin
{
   public:
	inline himan_plugin(){};

	inline virtual ~himan_plugin(){};

	virtual std::string ClassName() const = 0;

	virtual HPPluginClass PluginClass() const = 0;

	virtual HPVersionNumber Version() const = 0;

protected:
	logger itsLogger;
};

// the type of the class factory

typedef std::shared_ptr<himan_plugin> create_t();

}  // namespace plugin
}  // namespace himan

#endif /* HIMAN_PLUGIN_H */
