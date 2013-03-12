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

	cout	<< "Thread count:\t" <<  itsUsedThreadCount << endl
			<< "Cuda count:\t" << itsUsedCudaCount << endl
			<< "Elapsed time:\t" <<  elapsedTime << " microseconds" << endl
			<< "Fetching time:\t" << itsFetchingTime << " microseconds (" << static_cast<int> (100*static_cast<double> (itsFetchingTime)/static_cast<double> (elapsedTime)) << "%)" << endl
			<< "Process time:\t" << itsProcessingTime << " microseconds (" << static_cast<int> (100*static_cast<double> (itsProcessingTime)/static_cast<double> (elapsedTime)) << "%)" << endl
			<< "Writing time:\t" << itsWritingTime << " microseconds (" << static_cast<int> (100*static_cast<double> (itsWritingTime)/static_cast<double> (elapsedTime)) << "%)" << endl
			<< "Values:\t\t" << itsValueCount << endl
			<< "Missing values:\t" << itsMissingValueCount << endl
			<< "pps:\t\t" << 1000*1000*static_cast<double>(itsValueCount)/static_cast<double>(elapsedTime) << endl;

}

void statistics::AddToMissingCount(size_t theMissingCount)
{
	itsMissingValueCount += theMissingCount;
}

void statistics::AddToValueCount(size_t theValueCount)
{
	itsValueCount += theValueCount;
}

void statistics::AddToFetchingTime(size_t theFetchingTime)
{
	itsFetchingTime += theFetchingTime;
}

void statistics::AddToProcessingTime(size_t theProcessingTime)
{
	itsProcessingTime += theProcessingTime;
}

void statistics::AddToWritingTime(size_t theWritingTime)
{
	itsWritingTime += theWritingTime;
}

void statistics::Init()
{
	itsMissingValueCount = 0;
	itsValueCount = 0;
	itsUsedThreadCount = 0;
	itsUsedCudaCount = 0;
	itsFetchingTime = 0;
	itsProcessingTime = 0;
	itsWritingTime = 0;
}

void statistics::UsedThreadCount(size_t theUsedThreadCount)
{
	itsUsedThreadCount = theUsedThreadCount;
}

void statistics::UsedCudaCount(size_t theUsedCudaCount)
{
	itsUsedCudaCount = theUsedCudaCount;
}
