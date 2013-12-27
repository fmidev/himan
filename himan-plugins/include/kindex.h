/*
 * @file kindex.h
 *
 * @date Jan 23, 2012
 * @author: Aalto
 */

#ifndef KINDEX_H
#define KINDEX_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{

/**
 * @class kindex
 *
 * @brief Calculate k-indexs. 
 *
 */

class kindex : public compiled_plugin, private compiled_plugin_base
{
public:
    kindex();

    inline virtual ~kindex() {}

    kindex(const kindex& other) = delete;
    kindex& operator=(const kindex& other) = delete;

    virtual void Process(std::shared_ptr<const plugin_configuration> conf);

    virtual std::string ClassName() const
    {
        return "himan::plugin::kindex";
    }

    virtual HPPluginClass PluginClass() const
    {
        return kCompiled;
    }

    virtual HPVersionNumber Version() const
    {
        return HPVersionNumber(1, 0);
    }

private:
	virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);

};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<kindex> (new kindex());
}

} // namespace plugin
} // namespace himan

#endif /* KINDEX */
