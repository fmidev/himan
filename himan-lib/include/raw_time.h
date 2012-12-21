/*
 * raw_time.h
 *
 *  Created on: Dec 9, 2012
 *      Author: partio
 *
 * Wrapper for boost::posix_time.
 *
 */

#ifndef RAW_TIME_H
#define RAW_TIME_H

#include <boost/date_time.hpp>
#include "logger.h"
#include <NFmiMetTime.h>
#include <stdexcept>

namespace hilpee
{

class raw_time
{

	public:

		raw_time() {}
		raw_time(const std::string& theTime, const std::string& theTimeMask = "%Y-%m-%d %H:%M:%S");
		raw_time(const NFmiMetTime& theTime);

		std::string String(const std::string& theTimeMask = "%Y-%m-%d %H:%M:%S") const;

		std::ostream& Write(std::ostream& file) const;

		std::string ClassName() const
		{
			return "hilpee::raw_time";
		};

		HPVersionNumber Version() const
		{
			return HPVersionNumber(0, 1);
		}

		~raw_time() {}

		bool operator==(const raw_time& other);
		bool operator!=(const raw_time& other);

		bool Adjust(const std::string& theTimeType, int theValue);

		boost::posix_time::ptime RawTime() const
		{
			return itsDateTime;
		}

	private:

		boost::posix_time::ptime itsDateTime;
		std::string FormatTime(boost::posix_time::ptime theFormattedTime, const std::string& theTimeMask) const;

};


inline
std::ostream& operator<<(std::ostream& file, raw_time& ob)
{
	return ob.Write(file);
}

} // namespace hilpee

#endif /* RAW_TIME_H */
