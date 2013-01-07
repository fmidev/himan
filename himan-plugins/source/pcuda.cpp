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
{
    itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("pcuda"));
}

#ifndef HAVE_CUDA
bool pcuda::HaveCuda() const
{
    return false;
}
#else

bool pcuda::HaveCuda() const
{
    return !(DeviceCount() == 0);
}

int pcuda::DeviceCount() const
{
    // todo: error checking
    int devCount;
    cudaGetDeviceCount(&devCount);

    return devCount;
}

int pcuda::LibraryVersion() const
{
    // todo: error checking
    int ver;

    cudaDriverGetVersion(&ver);

    return ver;
}
/*
HPVersionNumber pcuda::ComputeCapability() const
{
	// todo: error checking

	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);

	return HPVersionNumber(boost::lexical_cast<unsigned short> (devProp.major), boost::lexical_cast<unsigned short> (devProp.minor));
}
*/
void pcuda::Capabilities() const
{
    int devCount;
    cudaGetDeviceCount(&devCount);

    cout << "**********" << endl;
    cout << "There are " << devCount << " CUDA devices" << endl;

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

    cout << "**********" << endl;
}
/*
  There are 1 CUDA devices
CUDA Device #0
Major revision number:         2
Minor revision number:         1
Name:                          NVS 4200M
Total global memory:           536018944
Total shared memory per block: 49152
Total registers per block:     32768
Warp size:                     32
Maximum memory pitch:          2147483647
Maximum threads per block:     1024
Maximum dimension 0 of block:  1024
Maximum dimension 1 of block:  1024
Maximum dimension 2 of block:  64
Maximum dimension 0 of grid:   65535
Maximum dimension 1 of grid:   65535
Maximum dimension 2 of grid:   65535
Clock rate:                    1480000
Total constant memory:         65536
Texture alignment:             512
Concurrent copy and execution: Yes
Number of multiprocessors:     1
Kernel execution timeout:      Yes
 */

#endif
