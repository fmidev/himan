/**
 * @file statistics.h
 *
 */

#ifndef STATISTICS_H
#define STATISTICS_H

#include "raw_time.h"
#include "timer.h"

#include <atomic>

namespace himan
{
class statistics
{
   public:
	friend class plugin_configuration;

	statistics();
	~statistics() = default;

	statistics(const statistics& other);
	statistics& operator=(const statistics& other) = delete;

	std::string ClassName() const
	{
		return "himan::statistics";
	}
	bool Start();
	bool Store();

	void AddToMissingCount(size_t theMissingCount);
	void AddToValueCount(size_t theValueCount);
	void AddToFetchingTime(int64_t theFetchingTime);
	void AddToProcessingTime(int64_t theProcessingTime);
	void AddToWritingTime(int64_t theWritingTime);
	void AddToInitTime(int64_t theInitTime);
	void AddToCacheMissCount(size_t theCacheMissCount);
	void AddToCacheHitCount(size_t theCacheHitCount);

	std::string Label() const;
	void Label(const std::string& theLabel);

	bool Enabled() const;

	void UsedThreadCount(short theThreadCount);
	void UsedGPUCount(short theGPUCount);

	int64_t FetchingTime() const;

   private:
	void Init();
	bool StoreToDatabase();
	bool StoreToFile();

	timer itsTimer;

	std::atomic<size_t> itsValueCount;
	std::atomic<size_t> itsMissingValueCount;
	std::atomic<int64_t> itsFetchingTime;
	std::atomic<int64_t> itsWritingTime;
	std::atomic<int64_t> itsProcessingTime;
	std::atomic<int64_t> itsInitTime;
	std::atomic<size_t> itsCacheMissCount;
	std::atomic<size_t> itsCacheHitCount;

	short itsUsedThreadCount;
	short itsUsedGPUCount;
};

}  // namespace himan

#endif /* STATISTICS_H */
