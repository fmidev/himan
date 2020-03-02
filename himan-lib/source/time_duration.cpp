#include "time_duration.h"
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/format.hpp>
#include <boost/regex.hpp>

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

std::string time_duration::String(const std::string& fmt) const
{
	// %H --> Total hours % 24
	// example: 33:00:00 --> %H = 9
	// %M --> Total minutes % 60
	// example: 01:15:00 --> %M = 15
	// %S --> Total seconds % 60
	// example: 00:10:30 --> %S = 30
	// %d --> Total days
	// example: 1 day 12:00:00 --> %d = 1
	// %h --> Total hours
	// example: 33:00:00 --> %H = 33
	// %m --> Total minutes
	// example: 01:15:00 --> %M = 75
	// %s --> Total seconds
	// example: 00:10:30 --> %S = 630

	auto ret = fmt;

	const static std::vector<std::pair<char, boost::regex>> regexs{
	    std::make_pair('H', boost::regex{R"(%([0-9]*)H)"}), std::make_pair('M', boost::regex{R"(%([0-9]*)M)"}),
	    std::make_pair('S', boost::regex{R"(%([0-9]*)S)"}), std::make_pair('d', boost::regex{R"(%([0-9]*)d)"}),
	    std::make_pair('h', boost::regex{R"(%([0-9]*)h)"}), std::make_pair('m', boost::regex{R"(%([0-9]*)m)"}),
	    std::make_pair('s', boost::regex{R"(%([0-9]*)s)"})};

	for (const auto& r : regexs)
	{
		const auto& re = r.second;
		int value;
		switch (r.first)
		{
			case 'H':
				value = static_cast<int>(Hours() % 24);
				break;
			case 'M':
				value = static_cast<int>(Minutes() % 60);
				break;
			case 'S':
				value = static_cast<int>(Seconds() % 60);
				break;
			case 'd':
				value = static_cast<int>(floor(static_cast<double>(Hours()) / 24.));
				break;
			case 'h':
				value = static_cast<int>(Hours());
				break;
			case 'm':
				value = static_cast<int>(Minutes());
				break;
			case 's':
				value = static_cast<int>(Seconds());
				break;
			default:
				break;
		}
		boost::smatch what;
		if (boost::regex_search(fmt, what, re))
		{
			ret = boost::regex_replace(ret, re, (boost::format("%" + std::string(what[1]) + "d") % value).str());
		}
	}

	return ret;
}
