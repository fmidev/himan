/**
 * @file forecast_time.cpp
 *
 */

#include "forecast_time.h"

using namespace himan;

forecast_time::forecast_time(const raw_time& theOriginDateTime, const raw_time& theValidDateTime)
    : itsOriginDateTime(theOriginDateTime), itsValidDateTime(theValidDateTime)
{
}

forecast_time::forecast_time(const raw_time& theOriginDateTime, const time_duration& theStep)
    : itsOriginDateTime(theOriginDateTime), itsValidDateTime(theOriginDateTime + theStep)
{
}

forecast_time::forecast_time(const std::string& theOriginDateTime, const std::string& theValidDateTime,
                             const std::string& theDateMask)
    : itsOriginDateTime(raw_time(theOriginDateTime, theDateMask)),
      itsValidDateTime(raw_time(theValidDateTime, theDateMask))
{
}

std::ostream& forecast_time::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;
	file << itsOriginDateTime;
	file << itsValidDateTime;

	return file;
}

bool forecast_time::operator==(const forecast_time& other) const
{
	if (this == &other)
	{
		return true;
	}

	return (itsOriginDateTime == other.itsOriginDateTime) && (itsValidDateTime == other.itsValidDateTime);
}

bool forecast_time::operator!=(const forecast_time& other) const
{
	return !(*this == other);
}
time_duration forecast_time::Step() const
{
	if (itsValidDateTime.Empty() || itsOriginDateTime.Empty())
	{
		return time_duration();
	}

	return time_duration(itsValidDateTime - itsOriginDateTime);
}

raw_time& forecast_time::OriginDateTime()
{
	return itsOriginDateTime;
}
const raw_time& forecast_time::OriginDateTime() const
{
	return itsOriginDateTime;
}
void forecast_time::OriginDateTime(const raw_time& theOriginDateTime)
{
	itsOriginDateTime = theOriginDateTime;
}
void forecast_time::OriginDateTime(const std::string& theOriginDateTime, const std::string& theDateMask)
{
	itsOriginDateTime = raw_time(theOriginDateTime, theDateMask);
}

raw_time& forecast_time::ValidDateTime()
{
	return itsValidDateTime;
}
const raw_time& forecast_time::ValidDateTime() const
{
	return itsValidDateTime;
}
void forecast_time::ValidDateTime(const std::string& theValidDateTime, const std::string& theDateMask)
{
	itsValidDateTime = raw_time(theValidDateTime, theDateMask);
}
void forecast_time::ValidDateTime(const raw_time& theValidDateTime)
{
	itsValidDateTime = theValidDateTime;
}
