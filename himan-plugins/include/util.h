/*
 * hilpee_util.h
 *
 *  Created on: Dec 1, 2012
 *      Author: partio
 *
 * All-purpose utility functions. A bit excess maybe to
 * create this as a separate plugin but its the easiest
 * and most logical way.
 *
 */

#ifndef UTIL_H
#define UTIL_H

#include "hilpee_common.h"
#include "auxiliary_plugin.h"

namespace hilpee
{
namespace plugin
{

class util : public auxiliary_plugin
{

	public:

		util();
		~util() {}

		virtual std::string ClassName() const
		{
			return "hilpee::plugin::util";
		};

		virtual HPPluginClass PluginClass() const
		{
			return kAuxiliary;
		};

		virtual HPVersionNumber Version() const
		{
			return HPVersionNumber(0, 1);
		}

		HPFileType FileType(const std::string& theFile) const;
		std::string MakeNeonsFileName(const hilpee::info& info) const;
};


#ifndef HILPEE_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<hilpee_plugin> create()
{
	return std::shared_ptr<util> (new util());
}

#endif /* HILPEE_AUXILIARY_INCLUDE */

} // namespace plugin
} // namespace hilpee

#endif /* UTIL_H */
