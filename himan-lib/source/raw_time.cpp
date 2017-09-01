/**
 * @file raw_time.cpp
 *
 */

#include "raw_time.h"
#include <mutex>

static std::mutex formatMutex;

using namespace himan;

raw_time::raw_time(const std::string& theDateTime, const std::string& theTimeMask)
{
	if (theTimeMask == "%Y-%m-%d %H:%M:%S")
	{
		FromSQLTime(theDateTime);
	}
	else if (theTimeMask == "%Y%m%d%H%M")
	{
		FromNeonsTime(theDateTime);
	}
	else
	{
		std::stringstream s(theDateTime);

		{
			std::lock_guard<std::mutex> lock(formatMutex);
			s.imbue(std::locale(s.getloc(), new boost::posix_time::time_input_facet(theTimeMask.c_str())));
		}

		s >> itsDateTime;
	}

	if (Empty())
	{
		throw std::runtime_error(ClassName() + ": Unable to create time from '" + theDateTime + "' with mask '" +
		                         theTimeMask + "'");
	}
}

raw_time::raw_time(const raw_time& other) : itsDateTime(other.itsDateTime) {}
raw_time& raw_time::operator=(const raw_time& other)
{
	itsDateTime = other.itsDateTime;

	return *this;
}

bool raw_time::operator==(const raw_time& other) const
{
	if (this == &other)
	{
		return true;
	}

	return (itsDateTime == other.itsDateTime);
}

bool raw_time::operator!=(const raw_time& other) const { return !(*this == other); }
raw_time::operator std::string() const { return ToNeonsTime(); }
std::string raw_time::String(const std::string& theTimeMask) const
{
	if (Empty())
	{
		throw std::runtime_error(ClassName() + ": input argument is not valid");
	}

	if (theTimeMask == "%Y-%m-%d %H:%M:%S")
	{
		return ToSQLTime();
	}
	else if (theTimeMask == "%Y%m%d%H%M")
	{
		return ToNeonsTime();
	}

	return FormatTime(theTimeMask);
}

std::string raw_time::FormatTime(const std::string& theTimeMask) const
{
	std::stringstream s;

	// https://stackoverflow.com/questions/11121454/c-why-is-my-date-parsing-not-threadsafe

	{
		std::lock_guard<std::mutex> lock(formatMutex);
		s.imbue(std::locale(s.getloc(), new boost::posix_time::time_facet(theTimeMask.c_str())));
	}

	s << itsDateTime;

	s.flush();

	return s.str();
}

bool raw_time::Adjust(HPTimeResolution timeResolution, int theValue)
{
	using namespace boost;

	if (timeResolution == kHourResolution)
	{
		posix_time::hours adjustment(theValue);

		itsDateTime = itsDateTime + adjustment;
	}
	else if (timeResolution == kMinuteResolution)
	{
		posix_time::minutes adjustment(theValue);

		itsDateTime = itsDateTime + adjustment;
	}
	else if (timeResolution == kYearResolution)
	{
		boost::gregorian::years adjustment(theValue);

		itsDateTime += adjustment;

		if (String("%m%d") == "0229")
		{
			itsDateTime += gregorian::date_duration(-1);
		}

		return true;
	}
	else if (timeResolution == kDayResolution)
	{
		gregorian::days adjustment(theValue);

		itsDateTime = itsDateTime + adjustment;
	}
	else
	{
		throw std::runtime_error(ClassName() + ": Invalid time adjustment unit: " +
		                         boost::lexical_cast<std::string>(timeResolution) + "'");
	}

	return true;
}

bool raw_time::Empty() const { return (itsDateTime == boost::posix_time::not_a_date_time); }
bool raw_time::IsLeapYear() const
{
	return boost::gregorian::gregorian_calendar::is_leap_year(itsDateTime.date().year());
}

std::ostream& raw_time::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;
	file << "__itsDateTime__ " << FormatTime("%Y-%m-%d %H:%M:%S") << std::endl;

	return file;
}

std::string raw_time::ToNeonsTime() const
{
	const auto& date = itsDateTime.date();
	const auto& time = itsDateTime.time_of_day();

	char fmt[13];
	snprintf(fmt, 13, "%04d%02d%02d%02d%02d", static_cast<int>(date.year()), static_cast<int>(date.month()),
	         static_cast<int>(date.day()), static_cast<int>(time.hours()), static_cast<int>(time.minutes()));

	return std::string(fmt);
}

void raw_time::FromNeonsTime(const std::string& neonsTime)
{
	const auto year = static_cast<unsigned short>(stoi(neonsTime.substr(0, 4)));
	const auto month = static_cast<unsigned short>(stoi(neonsTime.substr(4, 2)));
	const auto day = static_cast<unsigned short>(stoi(neonsTime.substr(6, 2)));
	const auto hour = static_cast<unsigned short>(stoi(neonsTime.substr(8, 2)));
	const auto minute = static_cast<unsigned short>(stoi(neonsTime.substr(10, 2)));

	using namespace boost;

	itsDateTime = posix_time::ptime(gregorian::date(year, month, day), posix_time::time_duration(hour, minute, 0, 0));
}

std::string raw_time::ToSQLTime() const
{
	const auto& date = itsDateTime.date();
	const auto& time = itsDateTime.time_of_day();

	char fmt[20];
	snprintf(fmt, 20, "%04d-%02d-%02d %02d:%02d:%02d", static_cast<int>(date.year()), static_cast<int>(date.month()),
	         static_cast<int>(date.day()), static_cast<int>(time.hours()), static_cast<int>(time.minutes()),
	         static_cast<int>(time.seconds()));

	return std::string(fmt);
}

void raw_time::FromSQLTime(const std::string& SQLTime) { itsDateTime = boost::posix_time::time_from_string(SQLTime); }
