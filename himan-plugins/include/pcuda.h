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

#define CUDA_CHECK(errarg)   __checkErrorFunc(errarg, __FILE__, __LINE__)
#define CHECK_ERROR_MSG(errstr) __checkErrMsgFunc(errstr, __FILE__, __LINE__)

inline void __checkErrorFunc(cudaError_t errarg, const char* file,
			     const int line)
{
    if(errarg) {
	fprintf(stderr, "Error at %s(%i)\n", file, line);
	exit(EXIT_FAILURE);
    }
}


inline void __checkErrMsgFunc(const char* errstr, const char* file,
			      const int line)
{
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
	fprintf(stderr, "Error: %s at %s(%i): %s\n",
		errstr, file, line, cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }
}


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

    void Capabilities() const;
    int LibraryVersion() const;
    int DeviceCount() const;
    HPVersionNumber ComputeCapability() const;
	
#endif
};

#ifdef HAVE_CUDA

inline int himan::plugin::pcuda::LibraryVersion() const
{
   // todo: error checking
    int ver;

    cudaDriverGetVersion(&ver);

    return ver;
}

inline void himan::plugin::pcuda::Capabilities() const
{
    int devCount;
    cudaGetDeviceCount(&devCount);

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
             << "Kernel execution timeout:      " << (devProp.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl;

    }
	std::cout << "#---------------------------------------------------#" << std::endl;

}
#endif

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
