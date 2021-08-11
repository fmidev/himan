/**
 * @file statistics.cpp
 */

#include "statistics.h"
#include <mutex>

using namespace std;
using namespace himan;

mutex summary;

statistics::statistics()
{
	itsMissingValueCount = 0;
	itsValueCount = 0;
	itsUsedThreadCount = 0;
	itsTotalTime = 0;
	itsFetchingTime = 0;
	itsProcessingTime = 0;
	itsWritingTime = 0;
	itsInitTime = 0;
	itsCacheHitCount = 0;
	itsCacheMissCount = 0;
}
statistics::statistics(const statistics& other) : itsUsedThreadCount(other.itsUsedThreadCount)

{
	itsValueCount.store(other.itsValueCount, std::memory_order_relaxed);
	itsMissingValueCount.store(other.itsMissingValueCount, std::memory_order_relaxed);
	itsTotalTime.store(other.itsTotalTime, std::memory_order_relaxed);
	itsFetchingTime.store(other.itsFetchingTime, std::memory_order_relaxed);
	itsWritingTime.store(other.itsWritingTime, std::memory_order_relaxed);
	itsProcessingTime.store(other.itsProcessingTime, std::memory_order_relaxed);
	itsInitTime.store(other.itsInitTime, std::memory_order_relaxed);
	itsCacheMissCount.store(other.itsCacheMissCount, std::memory_order_relaxed);
	itsCacheHitCount.store(other.itsCacheHitCount, std::memory_order_relaxed);
}

void statistics::AddToMissingCount(size_t theMissingCount)
{
	itsMissingValueCount += theMissingCount;
}
void statistics::AddToValueCount(size_t theValueCount)
{
	itsValueCount += theValueCount;
}
void statistics::AddToTotalTime(int64_t theTotalTime)
{
	itsTotalTime += theTotalTime;
}
void statistics::AddToFetchingTime(int64_t theFetchingTime)
{
	itsFetchingTime += theFetchingTime;
}
void statistics::AddToProcessingTime(int64_t theProcessingTime)
{
	itsProcessingTime += theProcessingTime;
}
void statistics::AddToWritingTime(int64_t theWritingTime)
{
	itsWritingTime += theWritingTime;
}
void statistics::AddToInitTime(int64_t theInitTime)
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
void statistics::UsedThreadCount(short theUsedThreadCount)
{
	itsUsedThreadCount = theUsedThreadCount;
}
int64_t statistics::FetchingTime() const
{
	return itsFetchingTime;
}
void statistics::AddToSummaryRecords(const summary_record& rec)
{
	lock_guard<mutex> lock(summary);
	itsSummaryRecords.push_back(rec);
}
vector<summary_record> statistics::SummaryRecords() const
{
	return itsSummaryRecords;
}
