/**
 * @file statistics.h
 *
 * @date Feb 8, 2013
 * @author partio
 */

#ifndef STATISTICS_H
#define STATISTICS_H

//#include "NFmiODBC.h"
#include "timer_factory.h"
#include "raw_time.h"

#if defined __GNUC__ && (__GNUC__ == 4 && __GNUC_MINOR__ < 5)
#include <cstdatomic>
#else
#include <atomic>
#endif


namespace himan
{

class statistics
{
public:
	friend class plugin_configuration;

	statistics();
	~statistics() {};

	statistics(const statistics& other) = delete;
	statistics& operator=(const statistics& other) = delete;

	std::string ClassName() const
	{
		return "himan::statistics";
	};

	HPVersionNumber Version() const
	{
		return HPVersionNumber(0, 1);
	}

	bool Start();
	bool Store();

	void AddToMissingCount(size_t theMissingCount);
	void AddToValueCount(size_t theValueCount);
	void AddToFetchingTime(size_t theFetchingTime);
	void AddToProcessingTime(size_t theProcessingTime);
	void AddToWritingTime(size_t theWritingTime);

	std::string Label() const;
	void Label(const std::string& theLabel);

	bool Enabled() const;

	void UsedThreadCount(size_t theThreadCount);
	void UsedCudaCount(size_t theCudaCount);

	void Write();
	
private:
	void Init();
	bool StoreToDatabase();
	bool StoreToFile();

	std::atomic<size_t> itsValueCount;
	std::atomic<size_t> itsMissingValueCount;
	std::atomic<size_t> itsFetchingTime;
	std::atomic<size_t> itsWritingTime;
	std::atomic<size_t> itsProcessingTime;
	std::shared_ptr<timer> itsTimer;
	size_t itsUsedThreadCount;
	size_t itsUsedCudaCount;
	

};

} // namespace himan

#endif /* STATISTICS_H */
