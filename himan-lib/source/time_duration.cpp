#include "time_duration.h"
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace himan;

time_duration::time_duration() : itsDuration(boost::posix_time::not_a_date_time)
{
}

time_duration::time_duration(const boost::posix_time::time_duration& dur) : itsDuration(dur)
{
}

time_duration::time_duration(HPTimeResolution theTimeResolution, long theTimeResolutionValue)
{
	using namespace boost::posix_time;

	switch (theTimeResolution)
	{
		case kDayResolution:
			itsDuration = hours(24 * theTimeResolutionValue);
			break;
		case kHourResolution:
			itsDuration = hours(theTimeResolutionValue);
			break;
		case kMinuteResolution:
			itsDuration = minutes(theTimeResolutionValue);
			break;
		default:
			throw std::runtime_error("Unsupported time resolution: " + HPTimeResolutionToString.at(theTimeResolution));
	}
}
time_duration time_duration::operator+(const time_duration& other) const
{
	return time_duration(itsDuration + other.itsDuration);
}
time_duration& time_duration::operator+=(const time_duration& other)
{
	itsDuration += other.itsDuration;
	return *this;
}
time_duration time_duration::operator-(const time_duration& other) const
{
	return time_duration(itsDuration - other.itsDuration);
}
time_duration& time_duration::operator-=(const time_duration& other)
{
	itsDuration -= other.itsDuration;
	return *this;
}
time_duration time_duration::operator*(int multiplier) const
{
	return time_duration(itsDuration * multiplier);
}
time_duration& time_duration::operator*=(int multiplier)
{
	itsDuration *= multiplier;
	return *this;
}
time_duration time_duration::operator/(int divisor) const
{
	return time_duration(itsDuration / divisor);
}
time_duration& time_duration::operator/=(int divisor)
{
	itsDuration /= divisor;
	return *this;
}

time_duration::time_duration(const std::string& SQLTimeInterval)
{
	itsDuration = boost::posix_time::duration_from_string(SQLTimeInterval);
}
bool time_duration::operator==(const time_duration& other) const
{
	return itsDuration == other.itsDuration;
}

bool time_duration::operator!=(const time_duration& other) const
{
	return !(*this == other);
}
bool time_duration::operator<=(const time_duration& other) const
{
	return itsDuration <= other.itsDuration;
}
bool time_duration::operator>=(const time_duration& other) const
{
	return itsDuration >= other.itsDuration;
}
bool time_duration::operator>(const time_duration& other) const
{
	return itsDuration > other.itsDuration;
}
bool time_duration::operator<(const time_duration& other) const
{
	return itsDuration < other.itsDuration;
}
time_duration::operator std::string() const
{
	return boost::posix_time::to_simple_string(itsDuration);
}

boost::posix_time::time_duration& time_duration::Raw()
{
	return itsDuration;
}

const boost::posix_time::time_duration& time_duration::Raw() const
{
	return itsDuration;
}

long time_duration::Hours() const
{
	return itsDuration.total_seconds() / 3600;
}

long time_duration::Minutes() const
{
	return itsDuration.total_seconds() / 60;
}

long time_duration::Seconds() const
{
	return itsDuration.total_seconds();
}

bool time_duration::Empty() const
{
	return itsDuration == boost::date_time::not_a_date_time;
}
std::ostream& time_duration::Write(std::ostream& file) const
{
	file << static_cast<std::string>(*this);

	return file;
}
