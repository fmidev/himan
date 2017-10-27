/**
 * @file aggregation.cpp
 *
 */

#include "aggregation.h"

using namespace himan;

aggregation::aggregation()
    : itsType(kUnknownAggregationType),
      itsTimeResolution(kUnknownTimeResolution),
      itsTimeResolutionValue(kHPMissingInt),
      itsFirstTimeValue(kHPMissingInt)
{
}

aggregation::aggregation(HPAggregationType theType, HPTimeResolution theTimeResolution, int theTimeResolutionValue,
                         int theFirstTimeValue)
    : itsType(theType),
      itsTimeResolution(theTimeResolution),
      itsTimeResolutionValue(theTimeResolutionValue),
      itsFirstTimeValue(theFirstTimeValue)
{
}

aggregation::aggregation(const aggregation& other)
    : itsType(other.itsType),
      itsTimeResolution(other.itsTimeResolution),
      itsTimeResolutionValue(other.itsTimeResolutionValue),
      itsFirstTimeValue(other.itsFirstTimeValue)
{
}

aggregation& aggregation::operator=(const aggregation& other)
{
	itsTimeResolutionValue = other.itsTimeResolutionValue;
	itsFirstTimeValue = other.itsFirstTimeValue;
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

	if (itsFirstTimeValue != other.itsFirstTimeValue)
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
void aggregation::FirstTimeValue(int theFirstTimeValue) { itsFirstTimeValue = theFirstTimeValue; }
int aggregation::FirstTimeValue() const { return itsFirstTimeValue; }
std::ostream& aggregation::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;

	file << "__itsType__ " << itsType << " (" << HPAggregationTypeToString.at(itsType) << ")" << std::endl;
	file << "__itsTimeResolution__ " << itsTimeResolution << " (" << HPTimeResolutionToString.at(itsTimeResolution)
	     << ")" << std::endl;
	file << "__itsTimeResolutionValue__ " << itsTimeResolutionValue << std::endl;
	file << "__itsFirstTimeValue__ " << itsFirstTimeValue << std::endl;

	return file;
}
