/**
 * @file aggregation.cpp
 * @author partio
 *
 * @date May 16, 2013, 2:12 PM
 */

#include "aggregation.h"

using namespace himan;

aggregation::aggregation()
	: itsAggregationType(kUnknownAggregationType)
	, itsTimeResolution(kUnknownTimeResolution)
	, itsTimeResolutionValue(kHPMissingInt)
{}

aggregation::aggregation(HPAggregationType theAggregationType, HPTimeResolution theTimeResolution, int theTimeResolutionValue)
	: itsAggregationType(theAggregationType)
	, itsTimeResolution(theTimeResolution)
	, itsTimeResolutionValue(theTimeResolutionValue)
{}

aggregation::aggregation(const aggregation& other)
	: itsAggregationType(other.itsAggregationType)
	, itsTimeResolution(other.itsTimeResolution)
	, itsTimeResolutionValue(other.itsTimeResolutionValue)
{}

aggregation& aggregation::operator=(const aggregation& other)
{
	itsTimeResolutionValue = other.itsTimeResolutionValue;
	itsTimeResolution = other.itsTimeResolution;
	itsAggregationType = other.itsAggregationType;

	return *this;
}

HPAggregationType aggregation::AggregationType() const
{
	return itsAggregationType;
}

void aggregation::AggregationType(HPAggregationType theAggregationType)
{
	itsAggregationType = theAggregationType;
}

HPTimeResolution aggregation::TimeResolution() const
{
	return itsTimeResolution;
}

void aggregation::TimeResolution(HPTimeResolution theTimeResolution)
{
	itsTimeResolution = theTimeResolution;
}

int aggregation::TimeResolutionValue() const
{
	return itsTimeResolutionValue;
}

void aggregation::TimeResolutionValue(int theTimeResolutionValue)
{
	itsTimeResolutionValue = theTimeResolutionValue;
}

std::ostream& aggregation::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;

	file << "__itsAggregationType__ " << itsAggregationType << std::endl;
	file << "__itsTimeResolution__ " << itsTimeResolution << std::endl;
	file << "__itsTimeResolutionValue__ " << itsTimeResolutionValue << std::endl;
	
	return file;
}