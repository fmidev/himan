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

namespace himan
{
namespace plugin
{

class grib : public auxiliary_plugin
{

public:

    grib();

    virtual ~grib() {}

    grib(const grib& other) = delete;
    grib& operator=(const grib& other) = delete;

    virtual std::string ClassName() const
    {
        return "himan::plugin::grib";
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

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<grib> (new grib());
}

#endif /* HIMAN_AUXILIARY_INCLUDE */

} // namespace plugin
} // namespace himan

#endif /* GRIB_H */
