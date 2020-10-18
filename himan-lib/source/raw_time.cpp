/**
 * @file raw_time.cpp
 *
 */

#include "raw_time.h"
#include <fmt/chrono.h>
#include <mutex>

static std::mutex formatMutex;

using namespace himan;

raw_time::raw_time(const boost::posix_time::ptime& ptime)
{
	itsDateTime = ptime;
}
raw_time::raw_time(const std::string& theDateTime, const std::string& theTimeMask)
{
	if (theTimeMask == "%Y-%m-%d %H:%M:%S")
	{
		FromSQLTime(theDateTime);
	}
	else if (theTimeMask == "%Y%m%d%H%M")
	{
		FromDatabaseTime(theDateTime);
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
		throw std::runtime_error(
		    fmt::format("{}: Unable to create time from '{}' with mask '{}'", ClassName(), theDateTime, theTimeMask));
	}
}

raw_time::raw_time(const raw_time& other) : itsDateTime(other.itsDateTime)
{
}
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

bool raw_time::operator!=(const raw_time& other) const
{
	return !(*this == other);
}
bool raw_time::operator<(const raw_time& other) const
{
	return itsDateTime < other.itsDateTime;
}
bool raw_time::operator>(const raw_time& other) const
{
	return itsDateTime > other.itsDateTime;
}
bool raw_time::operator>=(const raw_time& other) const
{
	return itsDateTime >= other.itsDateTime;
}
bool raw_time::operator<=(const raw_time& other) const
{
	return itsDateTime <= other.itsDateTime;
}
raw_time::operator std::string() const
{
	return ToDatabaseTime();
}
raw_time raw_time::operator+(const time_duration& adjustment) const
{
	return raw_time(itsDateTime + boost::posix_time::minutes(adjustment.Minutes()));
}
raw_time raw_time::operator-(const time_duration& adjustment) const
{
	return raw_time(itsDateTime - boost::posix_time::minutes(adjustment.Minutes()));
}
raw_time& raw_time::operator+=(const time_duration& adjustment)
{
	itsDateTime += boost::posix_time::minutes(adjustment.Minutes());
	return *this;
}
raw_time& raw_time::operator-=(const time_duration& adjustment)
{
	itsDateTime -= boost::posix_time::minutes(adjustment.Minutes());
	return *this;
}
time_duration raw_time::operator-(const raw_time& other) const
{
	return time_duration(itsDateTime - other.itsDateTime);
}

std::string raw_time::String(const std::string& theTimeMask) const
{
	if (Empty())
	{
		return "not_a_date_time";
	}

	if (theTimeMask == "%Y-%m-%d %H:%M:%S")
	{
		return ToSQLTime();
	}
	else if (theTimeMask == "%Y%m%d%H%M")
	{
		return ToDatabaseTime();
	}
	else if (theTimeMask == "%Y%m%d")
	{
		return ToDate();
	}
	else if (theTimeMask == "%H%M%S")
	{
		return ToTime();
	}

	return FormatTime(theTimeMask);
}

std::string raw_time::FormatTime(const std::string& theTimeMask) const
{
	try
	{
		return fmt::format(fmt::format("{{:{}}}", theTimeMask), boost::posix_time::to_tm(itsDateTime));
	}
	catch (const std::exception& e)
	{
		fmt::print("{}\n", e.what());
		himan::Abort();
	}
}

void raw_time::Adjust(HPTimeResolution timeResolution, int theValue)
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
	}
	else if (timeResolution == kDayResolution)
	{
		gregorian::days adjustment(theValue);

		itsDateTime = itsDateTime + adjustment;
	}
	else
	{
		throw std::runtime_error(fmt::format("{}: Invalid time adjustment unit: {}", timeResolution));
	}
}

bool raw_time::Empty() const
{
	return (itsDateTime == boost::posix_time::not_a_date_time);
}
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

std::string raw_time::ToDatabaseTime() const
{
	try
	{
		return fmt::format("{:%Y%m%d%H%M}", boost::posix_time::to_tm(itsDateTime));
	}
	catch (const std::exception& e)
	{
		fmt::print("{}\n", e.what());
		himan::Abort();
	}
}

void raw_time::FromDatabaseTime(const std::string& databaseTime)
{
	const auto year = static_cast<unsigned short>(stoi(databaseTime.substr(0, 4)));
	const auto month = static_cast<unsigned short>(stoi(databaseTime.substr(4, 2)));
	const auto day = static_cast<unsigned short>(stoi(databaseTime.substr(6, 2)));
	const auto hour = static_cast<unsigned short>(stoi(databaseTime.substr(8, 2)));
	const auto minute = static_cast<unsigned short>(stoi(databaseTime.substr(10, 2)));

	using namespace boost;

	itsDateTime = posix_time::ptime(gregorian::date(year, month, day), posix_time::time_duration(hour, minute, 0, 0));
}

std::string raw_time::ToSQLTime() const
{
	try
	{
		return fmt::format("{:%Y-%m-%d %H:%M:%S}", boost::posix_time::to_tm(itsDateTime));
	}
	catch (const std::exception& e)
	{
		fmt::print("{}\n", e.what());
		himan::Abort();
	}
}

std::string raw_time::ToDate() const
{
	try
	{
		// FMT_COMPILE with c++17
		return fmt::format("{:%Y%m%d}", boost::posix_time::to_tm(itsDateTime));
	}
	catch (const std::exception& e)
	{
		fmt::print("{}\n", e.what());
		himan::Abort();
	}
}

std::string raw_time::ToTime() const
{
	try
	{
		return fmt::format("{:%H%M%S}", boost::posix_time::to_tm(itsDateTime));
	}
	catch (const std::exception& e)
	{
		fmt::print("{}\n", e.what());
		himan::Abort();
	}
}

void raw_time::FromSQLTime(const std::string& SQLTime)
{
	itsDateTime = boost::posix_time::time_from_string(SQLTime);
}
