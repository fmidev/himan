/*
 * util.h
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

#include "auxiliary_plugin.h"

namespace himan
{
namespace plugin
{

class util : public auxiliary_plugin
{

	public:

		util();
		virtual ~util() {}

		util(const util& other) = delete;
		util& operator=(const util& other) = delete;

		virtual std::string ClassName() const
		{
			return "himan::plugin::util";
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
		std::string MakeNeonsFileName(const himan::info& info) const;
};


#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<util> (new util());
}

#endif /* HIMAN_AUXILIARY_INCLUDE */

} // namespace plugin
} // namespace himan

#endif /* UTIL_H */
