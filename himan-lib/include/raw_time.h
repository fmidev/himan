/**
 * @file raw_time.h
 *
 * @date Dec 9, 2012
 * @author partio
 *
 * @brief Wrapper for boost::posix_time.
 *
 */

#ifndef RAW_TIME_H
#define RAW_TIME_H

#include "himan_common.h"
#include <boost/date_time.hpp>

namespace himan
{

class logger;
class raw_time
{

public:
	friend class forecast_time;
	
    raw_time() : itsDateTime(boost::posix_time::not_a_date_time) {}
    raw_time(const std::string& theTime, const std::string& theTimeMask = "%Y-%m-%d %H:%M:%S");

    ~raw_time() {}
    raw_time(const raw_time& other);
    raw_time& operator=(const raw_time& other);
	operator std::string () const;
	
    std::string String(const std::string& theTimeMask = "%Y-%m-%d %H:%M:%S") const;

    std::ostream& Write(std::ostream& file) const;

    std::string ClassName() const
    {
        return "himan::raw_time";
    }

    bool operator==(const raw_time& other) const;
    bool operator!=(const raw_time& other) const;

    bool Adjust(HPTimeResolution timeResolution, int theValue);

    bool Empty() const;

private:

    std::string FormatTime(boost::posix_time::ptime theFormattedTime, const std::string& theTimeMask) const;

    boost::posix_time::ptime itsDateTime;

};

inline
std::ostream& operator<<(std::ostream& file, const raw_time& ob)
{
    return ob.Write(file);
}

} // namespace himan

#endif /* RAW_TIME_H */
