/**
 * @file aggregation.cpp
 *
 */

#include "aggregation.h"

using namespace himan;

aggregation::aggregation() : itsType(kUnknownAggregationType), itsTimeDuration()
{
}

aggregation::aggregation(HPAggregationType theAggregationType) : itsType(theAggregationType), itsTimeDuration()
{
}

aggregation::aggregation(HPAggregationType theAggregationType, const time_duration& theTimeDuration)
    : itsType(theAggregationType), itsTimeDuration(theTimeDuration)
{
}

bool aggregation::operator==(const aggregation& other) const
{
	if (this == &other)
	{
		return true;
	}

	if (itsType != other.itsType)
	{
		return false;
	}

	if (itsTimeDuration != other.itsTimeDuration)
	{
		return false;
	}

	return true;
}

bool aggregation::operator!=(const aggregation& other) const
{
	return !(*this == other);
}
HPAggregationType aggregation::Type() const
{
	return itsType;
}
void aggregation::Type(HPAggregationType theType)
{
	itsType = theType;
}
time_duration aggregation::TimeDuration() const
{
	return itsTimeDuration;
}
void aggregation::TimeDuration(const time_duration& theTimeDuration)
{
	itsTimeDuration = theTimeDuration;
}
std::ostream& aggregation::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;

	file << "__itsType__ " << itsType << " (" << HPAggregationTypeToString.at(itsType) << ")" << std::endl;
	file << "__itsTimeDuration__" << itsTimeDuration << std::endl;

	return file;
}
