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
		itsLogger->Warning("Failed to select device #" + boost::lexical_cast<string> (deviceId) + ", error: " + cudaGetErrorString(err));
		itsLogger->Warning("Has another CUDA process reserved the card?");
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
