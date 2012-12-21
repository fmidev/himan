/*
 * cache.h
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 */

#ifndef CACHE_H
#define CACHE_H

#include "auxiliary_plugin.h"
//#include "Cache.h"

namespace hilpee
{
namespace plugin
{

class cache : public auxiliary_plugin
{
	public:
		cache();

		virtual ~cache() {};

		virtual std::string ClassName() const
		{
			return "hilpee::plugin::cache";
		};

		virtual HPPluginClass PluginClass() const
		{
			return kAuxiliary;
		};

		virtual HPVersionNumber Version() const
		{
			return HPVersionNumber(0, 1);
		}

	private:

};

#ifndef HILPEE_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<hilpee_plugin> create()
{
	return std::shared_ptr<cache> (new cache());
}

#endif /* HILPEE_AUXILIARY_INCLUDE */

} // namespace plugin
} // namespace hilpee

#endif /* CACHE_H */
