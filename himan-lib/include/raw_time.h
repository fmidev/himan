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

#include <boost/date_time.hpp>
#include "logger.h"
#include <stdexcept>

namespace himan
{

class raw_time
{

public:

    raw_time() : itsDateTime(boost::posix_time::not_a_date_time) {}
    raw_time(const std::string& theTime, const std::string& theTimeMask = "%Y-%m-%d %H:%M:%S");

    ~raw_time() {}
    raw_time(const raw_time& other);
    raw_time& operator=(const raw_time& other);

    std::string String(const std::string& theTimeMask = "%Y-%m-%d %H:%M:%S") const;

    std::ostream& Write(std::ostream& file) const;

    std::string ClassName() const
    {
        return "himan::raw_time";
    };

    bool operator==(const raw_time& other);
    bool operator!=(const raw_time& other);

    bool Adjust(HPTimeResolution timeResolution, int theValue);

    boost::posix_time::ptime RawTime() const
    {
        return itsDateTime;
    }

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
