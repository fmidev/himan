/**
 * @file forecast_time.cpp
 *
 * @date Dec  1, 2012
 * @author partio
 */

#include "forecast_time.h"

using namespace himan;

forecast_time::forecast_time()
{
}

forecast_time::forecast_time(const raw_time& theOriginDateTime, const raw_time& theValidDateTime)
	: itsOriginDateTime(std::shared_ptr<raw_time> (new raw_time(theOriginDateTime)))
	, itsValidDateTime(std::shared_ptr<raw_time> (new raw_time(theValidDateTime)))
	, itsStepResolution(kHourResolution)
{
}

forecast_time::forecast_time(std::shared_ptr<raw_time> theOriginDateTime, std::shared_ptr<raw_time> theValidDateTime)
	: itsOriginDateTime(theOriginDateTime)
	, itsValidDateTime(theValidDateTime)
	, itsStepResolution(kHourResolution)
{
}

forecast_time::forecast_time(const std::string& theOriginDateTime,
							 const std::string& theValidDateTime,
							 const std::string& theDateMask)
	: itsOriginDateTime(std::shared_ptr<raw_time> (new raw_time(theOriginDateTime, theDateMask)))
	, itsValidDateTime(std::shared_ptr<raw_time> (new raw_time(theValidDateTime, theDateMask)))
	, itsStepResolution(kHourResolution)
{
}

forecast_time::forecast_time(const forecast_time& other)
	: itsOriginDateTime(std::shared_ptr<raw_time> (new raw_time(*other.itsOriginDateTime)))
	, itsValidDateTime(std::shared_ptr<raw_time> (new raw_time(*other.itsValidDateTime)))
	, itsStepResolution(other.itsStepResolution)
{
}

forecast_time& forecast_time::operator=(const forecast_time& other)
{
	itsOriginDateTime = std::shared_ptr<raw_time> (new raw_time(*other.itsOriginDateTime));
	itsValidDateTime = std::shared_ptr<raw_time> (new raw_time(*other.itsValidDateTime));
	itsStepResolution = other.itsStepResolution;

	return *this;
}

std::ostream& forecast_time::Write(std::ostream& file) const
{

	file << "<" << ClassName() << ">" << std::endl;
	file << *itsOriginDateTime;
	file << *itsValidDateTime;
	file << "__itsStepResolution__ " << itsStepResolution << std::endl;

	return file;
}

bool forecast_time::operator==(const forecast_time& other) const
{
	if (this == &other)
	{
		return true;
	}

	return ((*itsOriginDateTime == *other.itsOriginDateTime)
				&& (*itsValidDateTime == *other.itsValidDateTime)
				&& itsStepResolution == other.itsStepResolution);
}

bool forecast_time::operator!=(const forecast_time& other) const
{
	return !(*this == other);
}

int forecast_time::Step() const
{

	if (itsValidDateTime->RawTime() != boost::date_time::not_a_date_time && itsOriginDateTime->RawTime() != boost::date_time::not_a_date_time)
	{

		int step = kHPMissingInt;

		switch (itsStepResolution)
		{
		case kHourResolution:
			step = (itsValidDateTime->RawTime() - itsOriginDateTime->RawTime()).hours();
			break;

		case kMinuteResolution:
			step = (itsValidDateTime->RawTime() - itsOriginDateTime->RawTime()).total_seconds() / 60;
			break;

		default:
			throw std::runtime_error(ClassName() + ": unknown step resolution");
			break;
		}

		return step;

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


HPTimeResolution forecast_time::StepResolution() const
{
	return itsStepResolution;
}

void forecast_time::StepResolution(HPTimeResolution theStepResolution)
{
	itsStepResolution = theStepResolution;
}
