/**
 * @file raw_time.cpp
 *
 */

#include "raw_time.h"

using namespace himan;

raw_time::raw_time(const std::string& theDateTime, const std::string& theTimeMask)
{
	std::stringstream s(theDateTime);
	std::locale l(s.getloc(), new boost::posix_time::time_input_facet(theTimeMask.c_str()));

	s.imbue(l);

	s >> itsDateTime;

	if (itsDateTime == boost::date_time::not_a_date_time)
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

raw_time::operator std::string() const { return String("%Y%m%d%H%M"); }

std::string raw_time::String(const std::string& theTimeMask) const { return FormatTime(itsDateTime, theTimeMask); }

std::string raw_time::FormatTime(boost::posix_time::ptime theFormattedDateTime, const std::string& theTimeMask) const
{
	if (theFormattedDateTime == boost::date_time::not_a_date_time)
	{
		throw std::runtime_error(ClassName() + ": input argument is 'not-a-date-time'");
	}

	std::stringstream s;
	std::locale l(s.getloc(), new boost::posix_time::time_facet(theTimeMask.c_str()));

	s.imbue(l);

	s << theFormattedDateTime;

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

		if (gregorian::gregorian_calendar::is_leap_year(itsDateTime.date().year()))
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

std::ostream& raw_time::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;
	file << "__itsDateTime__ " << FormatTime(itsDateTime, "%Y-%m-%d %H:%M:%S") << std::endl;

	return file;
}
