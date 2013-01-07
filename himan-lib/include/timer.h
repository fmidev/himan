/*
 * timer.h
 *
 *  Created on: Dec 17, 2012
 *      Author: partio
 */

#ifndef TIMER_H
#define TIMER_H

#include <time.h>

namespace himan
{

class timer
{
public:
    timer() {}
    ~timer() {}

    inline void Start()
    {
        clock_gettime(CLOCK_REALTIME, &start_ts);
    }

    inline void Stop()
    {
        clock_gettime(CLOCK_REALTIME, &stop_ts);
    }

    inline long GetTime()
    {
        long start = static_cast<long> (start_ts.tv_sec*1000000000) + start_ts.tv_nsec;
        long stop =  static_cast<long> (stop_ts.tv_sec*1000000000) + stop_ts.tv_nsec;

        return (stop - start) / 1000; // us
    }

private:
    timespec start_ts;
    timespec stop_ts;

};

} // namespace himan

#endif /* TIMER_H */
