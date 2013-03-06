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

	void AddToMissingCount(size_t missingCount);
	void AddToValueCount(size_t valueCount);

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

	//std::string itsLabel;
    //std::unique_ptr<NFmiODBC> itsDatabase;
    
	std::atomic<size_t> itsValueCount;
	std::atomic<size_t> itsMissingValueCount;
	std::shared_ptr<timer> itsTimer;
	size_t itsUsedThreadCount;
	size_t itsUsedCudaCount;

};

} // namespace himan

#endif /* STATISTICS_H */
