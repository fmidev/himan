/**
 * @file timer.h
 *
 *
 * @brief Simple timer functionality
 */

#ifndef TIMER_H
#define TIMER_H

#include <time.h>

namespace himan
{
class timer
{
   public:
	timer() : start_ts{}, stop_ts{}
	{
	}
	timer(bool start) : start_ts{}, stop_ts{}
	{
		if (start)
			Start();
	}
	~timer()
	{
	}
	inline void Start()
	{
		clock_gettime(CLOCK_REALTIME, &start_ts);
	}
	inline void Stop()
	{
		clock_gettime(CLOCK_REALTIME, &stop_ts);
	}
	/**
	 * @return Elapsed time in milliseconds
	 */

	inline int64_t GetTime()
	{
		int64_t start = start_ts.tv_sec * 1000000000 + start_ts.tv_nsec;
		int64_t stop = stop_ts.tv_sec * 1000000000 + stop_ts.tv_nsec;

		return (stop - start) / 1000 / 1000;  // ms
	}

   private:
	timespec start_ts;
	timespec stop_ts;
};

}  // namespace himan

#endif /* TIMER_H */
