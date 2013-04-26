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

void statistics::AddToInitTime(size_t theInitTime)
{
	itsInitTime += theInitTime;
}

void statistics::AddToCacheMissCount(size_t theCacheMissCount)
{
	itsCacheMissCount += theCacheMissCount;
}

void statistics::AddToCacheHitCount(size_t theCacheHitCount)
{
	itsCacheHitCount += theCacheHitCount;
}

void statistics::Init()
{
	itsMissingValueCount = 0;
	itsValueCount = 0;
	itsUsedThreadCount = 0;
	itsUsedGPUCount = 0;
	itsFetchingTime = 0;
	itsProcessingTime = 0;
	itsWritingTime = 0;
	itsInitTime = 0;
	itsCacheHitCount = 0;
	itsCacheMissCount = 0;
}

void statistics::UsedThreadCount(size_t theUsedThreadCount)
{
	itsUsedThreadCount = theUsedThreadCount;
}

void statistics::UsedGPUCount(size_t theUsedGPUCount)
{
	itsUsedGPUCount = theUsedGPUCount;
}

size_t statistics::FetchingTime() const
{
	return itsFetchingTime;
}
