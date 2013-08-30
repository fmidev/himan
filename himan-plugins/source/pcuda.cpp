/*
 * pcuda.cpp
 *
 *  Created on: Dec 19, 2012
 *      Author: partio
 */

#include "logger_factory.h"
#include "pcuda.h"

#ifdef HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

using namespace std;
using namespace himan::plugin;

pcuda::pcuda()
#ifdef HAVE_CUDA
 : itsDeviceCount(kHPMissingInt)
#endif
{
    itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("pcuda"));
}

#ifndef HAVE_CUDA

bool pcuda::HaveCuda() const
{
    return false;
}

int himan::plugin::pcuda::DeviceCount() const
{
    return 0;
}
#else

int himan::plugin::pcuda::LibraryVersion() const
{
   // todo: error checking
    int ver;

    cudaDriverGetVersion(&ver);

    return ver;
}

int himan::plugin::pcuda::DeviceCount() const
{
    if (itsDeviceCount == kHPMissingInt)
    {
        cudaError_t err = cudaGetDeviceCount(&itsDeviceCount);

        if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver)
        {
            // No device or no driver present

    	    itsDeviceCount = 0;
        }
    }

    return itsDeviceCount;
}

void himan::plugin::pcuda::Capabilities() const
{
    int devCount = DeviceCount();

    if (devCount == 0)
    {
    	std::cout << "No CUDA devices found" << std::endl;
    	return;
    }

    std::cout << "#---------------------------------------------------#" << std::endl;
    std::cout << "CUDA library version " << LibraryVersion() << std::endl;
    std::cout << "There are " << devCount << " CUDA device(s)" << std::endl;

    // Iterate through devices
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        std::cout << "CUDA Device #" << i << std::endl;

        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);

        std::cout << "Major revision number:         " << devProp.major << std::endl
             << "Minor revision number:         " << devProp.minor << std::endl
             << "Name:                          " << devProp.name << std::endl
             << "Total global memory:           " << devProp.totalGlobalMem << std::endl
             << "Total shared memory per block: " << devProp.sharedMemPerBlock << std::endl
             << "Total registers per block:     " << devProp.regsPerBlock << std::endl
             << "Warp size:                     " << devProp.warpSize << std::endl
             << "Maximum memory pitch:          " << devProp.memPitch << std::endl
             << "Maximum threads per block:     " << devProp.maxThreadsPerBlock << std::endl;

        for (int i = 0; i < 3; ++i)
        {
            std::cout << "Maximum dimension " << i << " of block:  " << devProp.maxThreadsDim[i] << std::endl;
        }

        for (int i = 0; i < 3; ++i)
        {
            std::cout << "Maximum dimension " << i << " of grid:   " << devProp.maxGridSize[i] << std::endl;
        }

        std::cout << "Clock rate:                    " << devProp.clockRate << std::endl
             << "Total constant memory:         " << devProp.totalConstMem << std::endl
             << "Texture alignment:             " << devProp.textureAlignment << std::endl
             << "Concurrent copy and execution: " << (devProp.deviceOverlap ? "Yes" : "No") << std::endl
             << "Number of multiprocessors:     " << devProp.multiProcessorCount << std::endl
             << "Kernel execution timeout:      " << (devProp.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl << std::endl;

    }
	std::cout << "#---------------------------------------------------#" << std::endl;

}

bool pcuda::HaveCuda() const
{
    return !(DeviceCount() == 0);
}

himan::HPVersionNumber pcuda::ComputeCapability() const
{
	// todo: error checking

	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);

	return HPVersionNumber(boost::lexical_cast<unsigned short> (devProp.major), boost::lexical_cast<unsigned short> (devProp.minor));
}


#endif
