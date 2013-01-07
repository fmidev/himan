/*
 * himan_plugin.h
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 *
 * Top level class for all plugins. (Note: not for core-lib).
 */

#ifndef HIMAN_PLUGIN_H
#define HIMAN_PLUGIN_H

#include <string>
#include "himan_common.h"
#include "logger.h"

namespace himan
{

namespace plugin
{

/**
 * @interface Top class interface for all plugins
 */

class himan_plugin
{
public:

    inline himan_plugin() {};

    inline virtual ~himan_plugin() {};

    virtual std::string ClassName() const = 0;

    virtual HPPluginClass PluginClass() const = 0;

    virtual HPVersionNumber Version() const = 0;

protected:

    std::unique_ptr<logger> itsLogger;
};

// the type of the class factory

typedef std::shared_ptr<himan_plugin> create_t();

} // namespace plugin
} // namespace himan

#endif /* HIMAN_PLUGIN_H */
