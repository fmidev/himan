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

    bool HaveCuda() const;

    void Capabilities() const;

#ifdef HAVE_CUDA
    int LibraryVersion() const;
    int DeviceCount() const;
    HPVersionNumber ComputeCapability() const;
#endif
};

inline void pcuda::Capabilities() const
{
#ifdef HAVE_CUDA
    using std::cout;
    using std::endl;

    int devCount;
    cudaGetDeviceCount(&devCount);

    cout << "There are " << devCount << " CUDA device(s)" << endl;

    // Iterate through devices
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        cout << "CUDA Device #" << i << endl;

        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);

        cout << "Major revision number:         " << devProp.major << endl
             << "Minor revision number:         " << devProp.minor << endl
             << "Name:                          " << devProp.name << endl
             << "Total global memory:           " << devProp.totalGlobalMem << endl
             << "Total shared memory per block: " << devProp.sharedMemPerBlock << endl
             << "Total registers per block:     " << devProp.regsPerBlock << endl
             << "Warp size:                     " << devProp.warpSize << endl
             << "Maximum memory pitch:          " << devProp.memPitch << endl
             << "Maximum threads per block:     " << devProp.maxThreadsPerBlock << endl;

        for (int i = 0; i < 3; ++i)
        {
            cout << "Maximum dimension " << i << " of block:  " << devProp.maxThreadsDim[i] << endl;
        }

        for (int i = 0; i < 3; ++i)
        {
            cout << "Maximum dimension " << i << " of grid:   " << devProp.maxGridSize[i] << endl;
        }

        cout << "Clock rate:                    " << devProp.clockRate << endl
             << "Total constant memory:         " << devProp.totalConstMem << endl
             << "Texture alignment:             " << devProp.textureAlignment << endl
             << "Concurrent copy and execution: " << (devProp.deviceOverlap ? "Yes" : "No") << endl
             << "Number of multiprocessors:     " << devProp.multiProcessorCount << endl
             << "Kernel execution timeout:      " << (devProp.kernelExecTimeoutEnabled ? "Yes" : "No") << endl;

    }
#endif
}

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
