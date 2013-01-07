/*
 * pcuda.h
 *
 *  Created on: Dec 19, 2012
 *      Author: partio
 *
 */

#ifndef PCUDA_H
#define PCUDA_H

#include "auxiliary_plugin.h"
#include "himan_common.h"

namespace himan
{
namespace plugin
{

class pcuda : public auxiliary_plugin
{
public:
    pcuda();

    virtual ~pcuda() {};

    virtual std::string ClassName() const
    {
        return "himan::plugin::pcuda";
    }

    virtual HPPluginClass PluginClass() const
    {
        return kAuxiliary;
    }

    virtual HPVersionNumber Version() const
    {
        return HPVersionNumber(0, 1);
    }

    bool HaveCuda() const;

#ifdef HAVE_CUDA
    int LibraryVersion() const;
    int DeviceCount() const;
    HPVersionNumber ComputeCapability() const;
    void Capabilities() const;
#endif
};

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<pcuda> (new pcuda());
}

#endif /* HIMAN_AUXILIARY_INCLUDE */

}
}

#endif /* PCUDA_H */
