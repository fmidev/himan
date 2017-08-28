/**
 * @file forecast_time.cpp
 *
 */

#include "forecast_time.h"

using namespace himan;

forecast_time::forecast_time() {}
forecast_time::forecast_time(const raw_time& theOriginDateTime, const raw_time& theValidDateTime)
    : itsOriginDateTime(theOriginDateTime), itsValidDateTime(theValidDateTime), itsStepResolution(kHourResolution)
{
}

forecast_time::forecast_time(const std::string& theOriginDateTime, const std::string& theValidDateTime,
                             const std::string& theDateMask)
    : itsStepResolution(kHourResolution)
{
	itsOriginDateTime = raw_time(theOriginDateTime, theDateMask);
	itsValidDateTime = raw_time(theValidDateTime, theDateMask);
}

forecast_time::forecast_time(const forecast_time& other)
    : itsOriginDateTime(other.itsOriginDateTime),
      itsValidDateTime(other.itsValidDateTime),
      itsStepResolution(other.itsStepResolution)
{
}

forecast_time& forecast_time::operator=(const forecast_time& other)
{
	itsOriginDateTime = other.itsOriginDateTime;
	itsValidDateTime = other.itsValidDateTime;
	itsStepResolution = other.itsStepResolution;

	return *this;
}

std::ostream& forecast_time::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;
	file << itsOriginDateTime;
	file << itsValidDateTime;
	file << "__itsStepResolution__ " << itsStepResolution << std::endl;

	return file;
}

bool forecast_time::operator==(const forecast_time& other) const
{
	if (this == &other)
	{
		return true;
	}

	return ((itsOriginDateTime == other.itsOriginDateTime) && (itsValidDateTime == other.itsValidDateTime) &&
	        itsStepResolution == other.itsStepResolution);
}

bool forecast_time::operator!=(const forecast_time& other) const { return !(*this == other); }
int forecast_time::Step() const
{
	if (itsValidDateTime.itsDateTime != boost::date_time::not_a_date_time &&
	    itsOriginDateTime.itsDateTime != boost::date_time::not_a_date_time)
	{
		int step = kHPMissingInt;

		switch (itsStepResolution)
		{
			case kHourResolution:
				step = static_cast<int>((itsValidDateTime.itsDateTime - itsOriginDateTime.itsDateTime).hours());
				break;

			case kMinuteResolution:
				step = static_cast<int>((itsValidDateTime.itsDateTime - itsOriginDateTime.itsDateTime).total_seconds() /
				                        60);
				break;

			default:
				throw std::runtime_error(ClassName() + ": unknown step resolution");
				break;
		}

		return step;
	}

	return kHPMissingInt;
}

raw_time& forecast_time::OriginDateTime() { return itsOriginDateTime; }
const raw_time& forecast_time::OriginDateTime() const { return itsOriginDateTime; }
void forecast_time::OriginDateTime(const raw_time& theOriginDateTime) { itsOriginDateTime = theOriginDateTime; }
void forecast_time::OriginDateTime(const std::string& theOriginDateTime, const std::string& theDateMask)
{
	itsOriginDateTime = raw_time(theOriginDateTime, theDateMask);
}

raw_time& forecast_time::ValidDateTime() { return itsValidDateTime; }
const raw_time& forecast_time::ValidDateTime() const { return itsValidDateTime; }
void forecast_time::ValidDateTime(const std::string& theValidDateTime, const std::string& theDateMask)
{
	itsValidDateTime = raw_time(theValidDateTime, theDateMask);
}

HPTimeResolution forecast_time::StepResolution() const { return itsStepResolution; }
void forecast_time::StepResolution(HPTimeResolution theStepResolution) { itsStepResolution = theStepResolution; }
