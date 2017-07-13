/**
 * @file statistics.cpp
 */

#include "statistics.h"

using namespace std;
using namespace himan;

statistics::statistics() : itsTimer() { Init(); }
statistics::statistics(const statistics& other)
    : itsTimer(), itsUsedThreadCount(other.itsUsedThreadCount), itsUsedGPUCount(other.itsUsedGPUCount)

{
	itsValueCount.store(other.itsValueCount, std::memory_order_relaxed);
	itsMissingValueCount.store(other.itsMissingValueCount, std::memory_order_relaxed);
	itsFetchingTime.store(other.itsFetchingTime, std::memory_order_relaxed);
	itsWritingTime.store(other.itsWritingTime, std::memory_order_relaxed);
	itsProcessingTime.store(other.itsProcessingTime, std::memory_order_relaxed);
	itsInitTime.store(other.itsInitTime, std::memory_order_relaxed);
	itsCacheMissCount.store(other.itsCacheMissCount, std::memory_order_relaxed);
	itsCacheHitCount.store(other.itsCacheHitCount, std::memory_order_relaxed);
}

bool statistics::Start()
{
	itsTimer.Start();
	return true;
}

void statistics::AddToMissingCount(size_t theMissingCount) { itsMissingValueCount += theMissingCount; }
void statistics::AddToValueCount(size_t theValueCount) { itsValueCount += theValueCount; }
void statistics::AddToFetchingTime(size_t theFetchingTime) { itsFetchingTime += theFetchingTime; }
void statistics::AddToProcessingTime(size_t theProcessingTime) { itsProcessingTime += theProcessingTime; }
void statistics::AddToWritingTime(size_t theWritingTime) { itsWritingTime += theWritingTime; }
void statistics::AddToInitTime(size_t theInitTime) { itsInitTime += theInitTime; }
void statistics::AddToCacheMissCount(size_t theCacheMissCount) { itsCacheMissCount += theCacheMissCount; }
void statistics::AddToCacheHitCount(size_t theCacheHitCount) { itsCacheHitCount += theCacheHitCount; }
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

void statistics::UsedThreadCount(short theUsedThreadCount) { itsUsedThreadCount = theUsedThreadCount; }
void statistics::UsedGPUCount(short theUsedGPUCount) { itsUsedGPUCount = theUsedGPUCount; }
size_t statistics::FetchingTime() const { return itsFetchingTime; }
