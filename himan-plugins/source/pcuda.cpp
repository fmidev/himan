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

	if ((err = cudaSetDeviceFlags(cudaDeviceMapHost)) != cudaSuccess)
	{
		itsLogger->Error("Failed to set flags for device #" + boost::lexical_cast<string> (deviceId));
	}

	return true;	
}

#endif
