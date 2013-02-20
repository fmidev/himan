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

#else

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
