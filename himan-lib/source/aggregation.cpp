/**
 * @file aggregation.cpp
 *
 */

#include "aggregation.h"

using namespace himan;

aggregation::aggregation() : itsType(kUnknownAggregationType), itsTimeDuration(), itsTimeOffset()
{
}

aggregation::aggregation(HPAggregationType theAggregationType)
    : itsType(theAggregationType), itsTimeDuration(), itsTimeOffset()
{
}

aggregation::aggregation(HPAggregationType theAggregationType, const time_duration& theTimeDuration)
    : itsType(theAggregationType), itsTimeDuration(theTimeDuration), itsTimeOffset(theTimeDuration * -1)
{
}

aggregation::aggregation(HPAggregationType theAggregationType, const time_duration& theTimeDuration,
                         const time_duration& theTimeOffset)
    : itsType(theAggregationType), itsTimeDuration(theTimeDuration), itsTimeOffset(theTimeOffset)
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

	if (itsTimeOffset != other.itsTimeOffset)
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
time_duration aggregation::TimeOffset() const
{
	return itsTimeOffset;
}
void aggregation::TimeDuration(const time_duration& theTimeDuration)
{
	itsTimeDuration = theTimeDuration;
	if (itsTimeOffset.Empty())
	{
		itsTimeOffset = theTimeDuration * -1;
	}
}
void aggregation::TimeOffset(const time_duration& theTimeOffset)
{
	itsTimeOffset = theTimeOffset;
}

std::ostream& aggregation::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;

	file << "__itsType__ " << itsType << " (" << HPAggregationTypeToString.at(itsType) << ")" << std::endl;
	file << "__itsTimeDuration__" << itsTimeDuration << std::endl;
	file << "__itsTimeOffset__" << itsTimeOffset << std::endl;

	return file;
}
