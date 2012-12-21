/*
 * grib.h
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 */

#ifndef GRIB_H
#define GRIB_H

#include "auxiliary_plugin.h"
#include "NFmiGrib.h"
#include "search_options.h"

namespace hilpee
{
namespace plugin
{

class grib : public auxiliary_plugin
{

	public:

		grib();

		virtual ~grib() {}

		virtual std::string ClassName() const
		{
			return "hilpee::plugin::grib";
		};

		virtual HPPluginClass PluginClass() const
		{
			return kAuxiliary;
		};

		virtual HPVersionNumber Version() const
		{
			return HPVersionNumber(0, 1);
		}

		std::shared_ptr<NFmiGrib> Reader();

		std::vector<std::shared_ptr<info>> FromFile(const std::string& theInputFile, const search_options& options, bool theReadContents = true);

		bool ToFile(std::shared_ptr<info> theInfo, const std::string& theOutputFile, HPFileType theFileType, bool theActiveOnly);

	private:

		std::shared_ptr<NFmiGrib> itsGrib;

};

#ifndef HILPEE_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<hilpee_plugin> create()
{
	return std::shared_ptr<grib> (new grib());
}

#endif /* HILPEE_AUXILIARY_INCLUDE */

} // namespace plugin
} // namespace hilpee

#endif /* GRIB_H */
