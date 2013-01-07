/*
 * querydata.h
 *
 *  Created on: Nov 27, 2012
 *      Author: partio
 */

#ifndef QUERYDATA_H
#define QUERYDATA_H

#include "auxiliary_plugin.h"
#include "search_options.h"
#include "NFmiFastQueryInfo.h"

namespace himan
{
namespace plugin
{

class querydata : public auxiliary_plugin
{
public:
    querydata();

    virtual ~querydata() {}

    querydata(const querydata& other) = delete;
    querydata& operator=(const querydata& other) = delete;

    virtual std::string ClassName() const
    {
        return "himan::plugin::querydata";
    }

    virtual HPPluginClass PluginClass() const
    {
        return kAuxiliary;
    }

    virtual HPVersionNumber Version() const
    {
        return HPVersionNumber(0, 1);
    }

    std::shared_ptr<info> FromFile(const std::string& theInputFile, const search_options& options, bool theReadContents);
    bool ToFile(std::shared_ptr<info> theInfo, const std::string& theOutputFile, bool theActiveOnly);

private:

    NFmiTimeDescriptor CreateTimeDescriptor(std::shared_ptr<info> info, bool theActiveOnly);
    NFmiParamDescriptor CreateParamDescriptor(std::shared_ptr<info> info, bool theActiveOnly);
    NFmiHPlaceDescriptor CreateHPlaceDescriptor(std::shared_ptr<info> info);
    NFmiVPlaceDescriptor CreateVPlaceDescriptor(std::shared_ptr<info> info, bool theActiveOnly);

};

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<querydata> (new querydata());
}

#endif /* HIMAN_AUXILIARY_INCLUDE */

} // namespace plugin
} // namespace himan

#endif /* QUERYDATA_H */
