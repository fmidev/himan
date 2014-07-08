/*
 * pcuda.h
 *
 *  Created on: Dec 19, 2012
 *	  Author: partio
 *
 * Most of the functionality is in the header -- the himan executable does not necessarily
 * find all the necessary symbols if everything is defined in source.
 * This is very irritating since this means we need to link himan executable with cuda
 * libraries.
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
	pcuda() : itsDeviceCount(kHPMissingInt)
	{
		itsLogger = logger_factory::Instance()->GetLog("pcuda");
	}

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

	bool HaveCuda() const
	{
		return !(DeviceCount() == 0);
	}
	
	int DeviceCount() const;

#ifdef HAVE_CUDA

	void Capabilities() const;
	int LibraryVersion() const;
	bool SetDevice(int deviceId) const;
	int GetDevice() const;
	void Reset() const;
	
#endif

private:
	mutable int itsDeviceCount;
	
};

#ifndef HAVE_CUDA

inline
int pcuda::DeviceCount() const
{
	return 0;
}
#else

inline
void pcuda::Capabilities() const
{
	int devCount = DeviceCount();

	if (devCount == 0)
	{
		std::cout << "No CUDA devices found" << std::endl;
		return;
	}

	int runtimeVersion;

	CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));

	std::cout << "#----------------------------------------------#" << std::endl;
	std::cout << "CUDA library version " << LibraryVersion()/1000 << "." << (LibraryVersion()%100)/10 << std::endl;
	std::cout << "CUDA runtime version " << runtimeVersion/1000 << "." << (runtimeVersion%100)/10 << std::endl;
	std::cout << "There are " << devCount << " CUDA device(s)" << std::endl;
	std::cout << "#----------------------------------------------#" << std::endl;

	// Iterate through devices
	for (int i = 0; i < devCount; ++i)
	{
		// Get device properties
		std::cout << "CUDA Device #" << i << std::endl;

		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, i);

		std::cout << "Major revision number:\t\t" << devProp.major << std::endl
			 << "Minor revision number:\t\t" << devProp.minor << std::endl
			 << "Device name:\t\t\t" << devProp.name << std::endl
			 << "Total global memory:\t\t" << devProp.totalGlobalMem << std::endl
			 << "Total shared memory per block:\t" << devProp.sharedMemPerBlock << std::endl
			 << "Total registers per block:\t" << devProp.regsPerBlock << std::endl
			 << "Warp size:\t\t\t" << devProp.warpSize << std::endl
			 << "Maximum memory pitch:\t\t" << devProp.memPitch << std::endl
			 << "Maximum threads per block:\t" << devProp.maxThreadsPerBlock << std::endl;

		for (int i = 0; i < 3; ++i)
		{
			std::cout << "Maximum dimension " << i << " of block:\t" << devProp.maxThreadsDim[i] << std::endl;
		}

		for (int i = 0; i < 3; ++i)
		{
			std::cout << "Maximum dimension " << i << " of grid:\t" << devProp.maxGridSize[i] << std::endl;
		}

		std::cout << "Clock rate:\t\t\t" << devProp.clockRate << std::endl
			 << "Total constant memory:\t\t" << devProp.totalConstMem << std::endl
			 << "Texture alignment:\t\t" << devProp.textureAlignment << std::endl
			 << "Concurrent copy and execution:\t" << (devProp.deviceOverlap ? "Yes" : "No") << std::endl
			 << "Number of multiprocessors:\t" << devProp.multiProcessorCount << std::endl
			 << "Kernel execution timeout:\t" << (devProp.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl << std::endl;

	}
	std::cout << "#----------------------------------------------#" << std::endl;

}

inline
int pcuda::DeviceCount() const
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

inline
int pcuda::LibraryVersion() const
{
   // todo: error checking
	int ver;

	cudaDriverGetVersion(&ver);

	return ver;
}

inline
bool pcuda::SetDevice(int deviceId) const
{
	cudaError_t err;

	if ((err = cudaSetDevice(deviceId)) != cudaSuccess)
	{
		itsLogger->Error("Failed to select device #" + boost::lexical_cast<std::string> (deviceId) + ", error: " + cudaGetErrorString(err));
		itsLogger->Error("Has another CUDA process reserved the card?");
		return false;
	}

	return true;
}

inline
void pcuda::Reset() const
{
	cudaError_t err;

	if ((err = cudaDeviceReset()) != cudaSuccess)
	{
		itsLogger->Error("cudaDeviceReset() returned error (could be from an earlier async call)!");
	}
}

inline
int pcuda::GetDevice() const
{
	cudaError_t err;

	int ret = kHPMissingInt;

	if ((err = cudaGetDevice(&ret)) != cudaSuccess)
	{
		itsLogger->Error("cudaGetDevice() returned error");
		return kHPMissingInt;
	}

	return ret;
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
