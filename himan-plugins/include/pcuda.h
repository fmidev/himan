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

#ifdef HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

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

    /**
     * @brief Check if this server has cuda enabled devices
     */

    bool HaveCuda() const;
    int DeviceCount() const;

#ifdef HAVE_CUDA

    void Capabilities() const;
    int LibraryVersion() const;
    HPVersionNumber ComputeCapability() const;

#endif

private:
    mutable int itsDeviceCount;
	
};

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<pcuda> (new pcuda());
}

#endif /* HIMAN_AUXILIARY_INCLUDE */

} // namespace plugin
} // namespace himan

#endif /* PCUDA_H */
