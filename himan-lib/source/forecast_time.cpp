/*
 * forecast_time.cpp
 *
 *  Created on: Dec  1, 2012
 *      Author: partio
 */

#include "forecast_time.h"
#include "logger_factory.h"

using namespace hilpee;

forecast_time::forecast_time()
{
	itsLogger = logger_factory::Instance()->GetLog("forecast_time");
}

forecast_time::forecast_time(const raw_time& theOriginDateTime, const raw_time& theValidDateTime)
	: itsOriginDateTime(std::shared_ptr<raw_time> (new raw_time(theOriginDateTime)))
	, itsValidDateTime(std::shared_ptr<raw_time> (new raw_time(theValidDateTime)))
{
	itsLogger = logger_factory::Instance()->GetLog("forecast_time");
}
forecast_time::forecast_time(std::shared_ptr<raw_time> theOriginDateTime, std::shared_ptr<raw_time> theValidDateTime)
	: itsOriginDateTime(theOriginDateTime)
	, itsValidDateTime(theValidDateTime)
{
	itsLogger = logger_factory::Instance()->GetLog("forecast_time");
}

forecast_time::forecast_time(const std::string& theOriginDateTime,
                             const std::string& theValidDateTime,
                             const std::string& theDateMask)
	: itsOriginDateTime(std::shared_ptr<raw_time> (new raw_time(theOriginDateTime, theDateMask)))
	, itsValidDateTime(std::shared_ptr<raw_time> (new raw_time(theValidDateTime, theDateMask)))
{
	itsLogger = logger_factory::Instance()->GetLog("forecast_time");
}

std::ostream& forecast_time::Write(std::ostream& file) const
{

	file << "<" << ClassName() << " " << Version() << ">" << std::endl;
	file << *itsOriginDateTime;
	file << *itsValidDateTime;

	return file;
}

bool forecast_time::operator==(const forecast_time& other)
{
	return ((*itsOriginDateTime == *other.itsOriginDateTime) && (*itsValidDateTime == *other.itsValidDateTime));
}

bool forecast_time::operator!=(const forecast_time& other)
{
	return !(*this == other);
}

int forecast_time::Step() const
{

	if (itsValidDateTime->RawTime() != boost::date_time::not_a_date_time && itsOriginDateTime->RawTime() != boost::date_time::not_a_date_time)
	{
		return (itsValidDateTime->RawTime() - itsOriginDateTime->RawTime()).hours();
	}

	return kHPMissingInt;
}

std::shared_ptr<raw_time> forecast_time::OriginDateTime() const
{
	return itsOriginDateTime;
}

void forecast_time::OriginDateTime(std::shared_ptr<raw_time> theOriginDateTime)
{
	itsOriginDateTime = theOriginDateTime;
}

void forecast_time::OriginDateTime(std::string& theOriginDateTime, const std::string& theDateMask)
{
	itsOriginDateTime = std::shared_ptr<raw_time> (new raw_time(theOriginDateTime, theDateMask));
}

std::shared_ptr<raw_time> forecast_time::ValidDateTime() const
{
	return itsValidDateTime;
}

void forecast_time::ValidDateTime(std::shared_ptr<raw_time> theValidDateTime)
{
	itsValidDateTime = theValidDateTime;
}

void forecast_time::ValidDateTime(std::string& theValidDateTime, const std::string& theDateMask)
{
	itsValidDateTime = std::shared_ptr<raw_time> (new raw_time(theValidDateTime, theDateMask));
}
