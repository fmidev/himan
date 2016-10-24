/**
 * @file aggregation.cpp
 *
 */

#include "aggregation.h"

using namespace himan;

aggregation::aggregation()
    : itsType(kUnknownAggregationType), itsTimeResolution(kUnknownTimeResolution), itsTimeResolutionValue(kHPMissingInt)
{
}

aggregation::aggregation(HPAggregationType theType, HPTimeResolution theTimeResolution, int theTimeResolutionValue)
    : itsType(theType), itsTimeResolution(theTimeResolution), itsTimeResolutionValue(theTimeResolutionValue)
{
}

aggregation::aggregation(const aggregation& other)
    : itsType(other.itsType),
      itsTimeResolution(other.itsTimeResolution),
      itsTimeResolutionValue(other.itsTimeResolutionValue)
{
}

aggregation& aggregation::operator=(const aggregation& other)
{
	itsTimeResolutionValue = other.itsTimeResolutionValue;
	itsTimeResolution = other.itsTimeResolution;
	itsType = other.itsType;

	return *this;
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

	if (itsTimeResolution != other.itsTimeResolution)
	{
		return false;
	}

	if (itsTimeResolutionValue != other.itsTimeResolutionValue)
	{
		return false;
	}

	return true;
}

bool aggregation::operator!=(const aggregation& other) const { return !(*this == other); }

HPAggregationType aggregation::Type() const { return itsType; }

void aggregation::Type(HPAggregationType theType) { itsType = theType; }

HPTimeResolution aggregation::TimeResolution() const { return itsTimeResolution; }

void aggregation::TimeResolution(HPTimeResolution theTimeResolution) { itsTimeResolution = theTimeResolution; }

int aggregation::TimeResolutionValue() const { return itsTimeResolutionValue; }

void aggregation::TimeResolutionValue(int theTimeResolutionValue) { itsTimeResolutionValue = theTimeResolutionValue; }

std::ostream& aggregation::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;

	file << "__itsType__ " << itsType << " (" << HPAggregationTypeToString.at(itsType) << ")" << std::endl;
	file << "__itsTimeResolution__ " << itsTimeResolution << " (" << HPTimeResolutionToString.at(itsTimeResolution)
	     << ")" << std::endl;
	file << "__itsTimeResolutionValue__ " << itsTimeResolutionValue << std::endl;

	return file;
}
