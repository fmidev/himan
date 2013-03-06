/**
 * @file statistics.cpp
 */

#include "statistics.h"

using namespace std;
using namespace himan;

statistics::statistics()
{
	itsTimer = shared_ptr<timer> (timer_factory::Instance()->GetTimer());
	Init();
}

bool statistics::Start()
{
	itsTimer->Start();
	return true;
}

void statistics::Write()
{
	itsTimer->Stop();

	size_t elapsedTime = itsTimer->GetTime();

	cout << "Thread count:\t" <<  itsUsedThreadCount << endl;
	cout << "Cuda count:\t" << itsUsedCudaCount << endl;
	cout << "Elapsed time:\t" <<  elapsedTime << " microseconds" << endl;
	cout << "Values:\t\t" << itsValueCount << endl;
	cout << "Missing values:\t" << itsMissingValueCount << endl;

	cout << "pps:\t\t" << 1000*1000*static_cast<double>(itsValueCount)/static_cast<double>(elapsedTime) << endl;

}

void statistics::AddToMissingCount(size_t missingCount)
{
	itsMissingValueCount += missingCount;
}

void statistics::AddToValueCount(size_t valueCount)
{
	itsValueCount += valueCount;
}

void statistics::Init()
{
	itsMissingValueCount = 0;
	itsValueCount = 0;
	itsUsedThreadCount = 0;
	itsUsedCudaCount = 0;
}

void statistics::UsedThreadCount(size_t theUsedThreadCount)
{
	itsUsedThreadCount = theUsedThreadCount;
}

void statistics::UsedCudaCount(size_t theUsedCudaCount)
{
	itsUsedCudaCount = theUsedCudaCount;
}
