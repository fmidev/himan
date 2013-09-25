/*
 * pcuda.cpp
 *
 *  Created on: Dec 19, 2012
 *      Author: partio
 */

#include "logger_factory.h"
#include "pcuda.h"

using namespace std;
using namespace himan::plugin;

pcuda::pcuda()
#ifdef HAVE_CUDA
 : itsDeviceCount(kHPMissingInt)
#endif
{
    itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("pcuda"));
}

#ifdef HAVE_CUDA

himan::HPVersionNumber pcuda::ComputeCapability() const
{
	// todo: error checking

	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);

	return HPVersionNumber(boost::lexical_cast<unsigned short> (devProp.major), boost::lexical_cast<unsigned short> (devProp.minor));
}

bool pcuda::SetDevice(int deviceId) const
{
	cudaError_t err;

	if ((err = cudaSetDevice(deviceId)) != cudaSuccess)
	{
		itsLogger->Error("Failed to choose device #" + boost::lexical_cast<string> (deviceId));
		return false;
	}

	// cudaDeviceMapHost is needed for zero copy memory
	
	if ((err = cudaSetDeviceFlags(cudaDeviceMapHost)) != cudaSuccess)
	{
		itsLogger->Error("Failed to set flags for device #" + boost::lexical_cast<string> (deviceId));
		itsLogger->Error("Return code: " + boost::lexical_cast<string> (err));
		return false;
	}

	return true;	
}

void pcuda::Reset() const
{
	cudaError_t err;

	if ((err = cudaDeviceReset()) != cudaSuccess)
	{
		itsLogger->Error("cudaDeviceReset() returned error (probably from earlier async call)!");
	}
}

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
