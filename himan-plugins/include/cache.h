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

namespace himan
{
namespace plugin
{

class cache : public auxiliary_plugin
{
	public:
		cache();

		virtual ~cache() {};

		cache(const cache& other) = delete;
		cache& operator=(const cache& other) = delete;

		virtual std::string ClassName() const
		{
			return "himan::plugin::cache";
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

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<cache> (new cache());
}

#endif /* HIMAN_AUXILIARY_INCLUDE */

} // namespace plugin
} // namespace himan

#endif /* CACHE_H */
