/*
 * cache.h
 *
 *  Created on: Nov 20, 2012
 *      Author: partio, perämäki
 */

#ifndef CACHE_H
#define CACHE_H

#include "auxiliary_plugin.h"
//#include <boost/optional.hpp>
#include <boost/thread.hpp>
#include "search_options.h"
//#include "Cache.h"

namespace himan
{
namespace plugin
{

//class cachePool;

class cache : public auxiliary_plugin
{    

public:
 
    ~cache() { delete itsInstance; }

    cache(const cache& other) = delete;
    cache& operator=(const cache& other) = delete;

    static cache* Instance();
    bool Find(const std::string& uniqueName);
    std::string UniqueName(const search_options& options);
    void Insert(const search_options& options, std::vector<std::shared_ptr<himan::info>> infos);
    std::shared_ptr<himan::info> GetInfo(const search_options& options);
    

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

    cache();

    
    std::map<std::string, std::shared_ptr<himan::info>> itsCache;
    static cache* itsInstance;

};


#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<cache> (cache::Instance());
}

#endif /* HIMAN_AUXILIARY_INCLUDE */

} // namespace plugin
} // namespace himan

#endif /* CACHE_H */
