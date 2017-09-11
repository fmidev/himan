/**
 * @file modifier.cpp
 */

#include "modifier.h"
#include "NFmiInterpolation.h"
#include "numerical_functions.h"

using namespace himan;
using namespace himan::numerical_functions;

#ifdef DEBUG
#include "util.h"
#include <iostream>
#endif
#include <iostream>

const double DEFAULT_MAXIMUM = 1e38;
const double DEFAULT_MINIMUM = -1e38;

double ExactEdgeValue(double theHeight, double theValue, double thePreviousHeight, double thePreviousValue,
                      double theLimit)
{
	if (IsMissing(thePreviousValue) || IsMissing(thePreviousHeight))
	{
		return theValue;
	}

	double ret = interpolation::Linear(theLimit, thePreviousHeight, theHeight, thePreviousValue, theValue);
	return ret;
}

modifier::modifier()
    : itsFindNthValue(1),
      itsIndex(0),
      itsModifierType(kUnknownModifierType),
      itsHeightInMeters(true),
      itsGridsProcessed(0)
{
}

modifier::modifier(HPModifierType theModifierType)
    : itsFindNthValue(1), itsIndex(0), itsModifierType(theModifierType), itsHeightInMeters(true), itsGridsProcessed(0)
{
}

const std::vector<double>& modifier::Result() const
{
	assert(itsResult.size());

	return itsResult;
}

bool modifier::CalculationFinished() const
{
	if (itsResult.size() > 0 &&
	    static_cast<size_t>(count(itsOutOfBoundHeights.begin(), itsOutOfBoundHeights.end(), true)) == itsResult.size())
	{
		return true;
	}

	return false;
}

void modifier::Clear(double fillValue)
{
	std::fill(itsResult.begin(), itsResult.end(), fillValue);
	std::fill(itsPreviousValue.begin(), itsPreviousValue.end(), fillValue);
	std::fill(itsPreviousHeight.begin(), itsPreviousHeight.end(), fillValue);
	std::fill(itsOutOfBoundHeights.begin(), itsOutOfBoundHeights.end(), false);
}

std::vector<double> modifier::FindValue() const { return itsFindValue; }
std::vector<double> modifier::LowerHeight() const { return itsLowerHeight; }
std::vector<double> modifier::UpperHeight() const { return itsUpperHeight; }
void modifier::FindValue(const std::vector<double>& theFindValue)
{
	itsFindValue = theFindValue;

	// If Find values have missing values we can't process those grid points

	itsOutOfBoundHeights.resize(itsFindValue.size(), false);

	for (size_t i = 0; i < itsFindValue.size(); i++)
	{
		if (IsMissing(itsFindValue[i]))
		{
			itsOutOfBoundHeights[i] = true;
		}
	}
#ifdef EXTRADEBUG
	util::DumpVector(itsFindValue);
#endif
}

void modifier::LowerHeight(const std::vector<double>& theLowerHeight)
{
	itsLowerHeight = theLowerHeight;

	// If height limits have missing values we can't process those grid points

	itsOutOfBoundHeights.resize(itsLowerHeight.size(), false);

	for (size_t i = 0; i < itsLowerHeight.size(); i++)
	{
		if (IsMissing(itsLowerHeight[i]))
		{
			itsOutOfBoundHeights[i] = true;
		}
	}
#ifdef EXTRADEBUG
	util::DumpVector(itsLowerHeight);
#endif
}

void modifier::UpperHeight(const std::vector<double>& theUpperHeight)
{
	itsUpperHeight = theUpperHeight;

	// If height limits have missing values we can't process those grid points

	itsOutOfBoundHeights.resize(itsUpperHeight.size(), false);

	for (size_t i = 0; i < itsUpperHeight.size(); i++)
	{
		if (IsMissing(itsUpperHeight[i]))
		{
			itsOutOfBoundHeights[i] = true;
		}
	}
#ifdef EXTRADEBUG
	util::DumpVector(itsUpperHeight);
#endif
}

size_t modifier::FindNth() const { return itsFindNthValue; }
void modifier::FindNth(size_t theNth) { itsFindNthValue = theNth; }
double modifier::Value() const
{
	assert(itsIndex < itsResult.size());

	return itsResult[itsIndex];
}

void modifier::Value(double theValue)
{
	assert(itsIndex < itsResult.size());

	itsResult[itsIndex] = theValue;
}

void modifier::Init(const std::vector<double>& theData, const std::vector<double>& theHeights)
{
	if (itsResult.size() == 0)
	{
		assert(theData.size() == theHeights.size());

		itsResult.resize(theData.size(), MissingDouble());
		itsOutOfBoundHeights.resize(theData.size(), false);
		itsPreviousValue.resize(itsResult.size(), MissingDouble());
		itsPreviousHeight.resize(itsResult.size(), MissingDouble());

		InitializeHeights();

#ifdef EXTRADEBUG
		util::DumpVector(itsLowerHeight);
		util::DumpVector(itsUpperHeight);
#endif
	}
}

bool modifier::Evaluate(double theValue, double theHeight, double thePreviousValue, double thePreviousHeight)
{
	assert(itsIndex < itsOutOfBoundHeights.size());
	assert(itsIndex < itsLowerHeight.size());
	assert(itsIndex < itsUpperHeight.size());

	if (itsOutOfBoundHeights[itsIndex])
	{
		return false;
	}

	const double lowerLimit = itsLowerHeight[itsIndex];
	const double upperLimit = itsUpperHeight[itsIndex];

	assert((itsHeightInMeters && lowerLimit <= upperLimit) || (!itsHeightInMeters && lowerLimit >= upperLimit));

	if (IsMissing(theHeight) || IsMissing(theValue))
	{
		return false;
	}

	// Iff the branching caused by all the conditions where height type is checked
	// is deteriorating performance (this should be verified!), we could create a new
	// height class/interface and just use inheritance for meter/pascal variants.

	if (itsHeightInMeters)
	{
		if (theHeight < lowerLimit)
		{
			// height is below given height range, do not cancel calculation yet
			return false;
		}
		else if (theHeight > upperLimit && thePreviousHeight > upperLimit)
		{
			// Safely above upper limit
			itsOutOfBoundHeights[itsIndex] = true;
			return false;
		}
	}
	else
	{
		if (theHeight > lowerLimit)
		{
			return false;
		}
		else if (theHeight < upperLimit && thePreviousHeight < upperLimit)
		{
			itsOutOfBoundHeights[itsIndex] = true;
			return false;
		}
	}

	return true;
}

void modifier::Process(const std::vector<double>& theData, const std::vector<double>& theHeights)
{
	Init(theData, theHeights);

	// assert(itsResult.size() == theData.size() && itsResult.size() == theHeights.size());

	for (itsIndex = 0; itsIndex < theData.size(); itsIndex++)
	{
		double theValue = theData[itsIndex];
		double theHeight = theHeights[itsIndex];

		double thePreviousValue = itsPreviousValue[itsIndex];
		double thePreviousHeight = itsPreviousHeight[itsIndex];

		if (!IsMissing(theValue) && !IsMissing(theHeight))
		{
			// If vertical profile has gaps (missing values or heights)
			// those should not be included as previous values because
			// by skipping them we can still save the calculation
			// (even though the value is more inprecise)

			itsPreviousValue[itsIndex] = theValue;
			itsPreviousHeight[itsIndex] = theHeight;
		}

		if (!Evaluate(theValue, theHeight, thePreviousValue, thePreviousHeight))
		{
			continue;
		}

		Calculate(theValue, theHeight, thePreviousValue, thePreviousHeight);
	}

	itsGridsProcessed++;
}

size_t modifier::HeightsCrossed() const
{
	return static_cast<size_t>(count(itsOutOfBoundHeights.begin(), itsOutOfBoundHeights.end(), true));
}

HPModifierType modifier::Type() const { return itsModifierType; }
bool modifier::HeightInMeters() const { return itsHeightInMeters; }
void modifier::HeightInMeters(bool theHeightInMeters) { itsHeightInMeters = theHeightInMeters; }
void modifier::InitializeHeights()
{
	// Absurd default limits if user has not specified any limits

	assert(itsResult.size());

	double min = (itsHeightInMeters) ? DEFAULT_MINIMUM : DEFAULT_MAXIMUM;
	double max = (itsHeightInMeters) ? DEFAULT_MAXIMUM : DEFAULT_MINIMUM;

	if (itsLowerHeight.empty())
	{
		itsLowerHeight.resize(itsResult.size(), min);
	}

	if (itsUpperHeight.empty())
	{
		itsUpperHeight.resize(itsResult.size(), max);
	}
}

std::ostream& modifier::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;

	file << "__itsFindNthValue__ " << itsFindNthValue << std::endl;
	file << "__itsIndex__ " << itsIndex << std::endl;
	file << "__itsResult__ size " << itsResult.size() << std::endl;
	file << "__itsFindValue__ size " << itsFindValue.size() << std::endl;
	file << "__itsLowerHeight__ size " << itsLowerHeight.size() << std::endl;
	file << "__itsUpperHeight__ size " << itsUpperHeight.size() << std::endl;
	file << "__itsOutOfBoundHeights__ size " << itsOutOfBoundHeights.size() << std::endl;

	return file;
}

bool modifier::EnteringHeightZone(double theHeight, double thePreviousHeight, double lowerLimit) const
{
	if (itsHeightInMeters)
	{
		return (!IsMissing(thePreviousHeight) && lowerLimit != DEFAULT_MINIMUM && theHeight >= lowerLimit &&
		        thePreviousHeight < lowerLimit);
	}
	else
	{
		return (!IsMissing(thePreviousHeight) && lowerLimit != DEFAULT_MAXIMUM && theHeight <= lowerLimit &&
		        thePreviousHeight > lowerLimit);
	}
}

bool modifier::LeavingHeightZone(double theHeight, double thePreviousHeight, double upperLimit) const
{
	if (itsHeightInMeters)
	{
		return (upperLimit != DEFAULT_MAXIMUM && theHeight >= upperLimit && thePreviousHeight < upperLimit);
	}
	else
	{
		return (upperLimit != DEFAULT_MINIMUM && theHeight <= upperLimit && thePreviousHeight > upperLimit);
	}
}

bool modifier::BetweenLevels(double theHeight, double thePreviousHeight, double lowerLimit, double upperLimit) const
{
	if (itsHeightInMeters)
	{
		return (thePreviousHeight <= lowerLimit && theHeight >= lowerLimit && thePreviousHeight <= upperLimit &&
		        theHeight >= upperLimit);
	}
	else
	{
		return (thePreviousHeight >= lowerLimit && theHeight <= lowerLimit && thePreviousHeight >= upperLimit &&
		        theHeight <= upperLimit);
	}
}

/* ----------------- */

void modifier_max::Calculate(double theValue, double theHeight, double thePreviousValue, double thePreviousHeight)
{
	double lowerLimit = itsLowerHeight[itsIndex];
	double upperLimit = itsUpperHeight[itsIndex];

	if (BetweenLevels(theHeight, thePreviousHeight, lowerLimit, upperLimit))
	{
		auto exactLower = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, lowerLimit);
		auto exactUpper = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, upperLimit);
		theValue = fmax(exactLower, exactUpper);
	}
	else if (EnteringHeightZone(theHeight, thePreviousHeight, lowerLimit))
	{
		double exactLower = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, lowerLimit);
		theValue = fmax(exactLower, theValue);
	}
	else if (LeavingHeightZone(theHeight, thePreviousHeight, upperLimit))
	{
		double exactUpper = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, upperLimit);
		theValue = fmax(exactUpper, theValue);
		itsOutOfBoundHeights[itsIndex] = true;
	}

	if (IsMissing(Value()) || theValue > Value())
	{
		Value(theValue);
	}
}

/* ----------------- */

void modifier_min::Calculate(double theValue, double theHeight, double thePreviousValue, double thePreviousHeight)
{
	double lowerLimit = itsLowerHeight[itsIndex];
	double upperLimit = itsUpperHeight[itsIndex];

	if (BetweenLevels(theHeight, thePreviousHeight, lowerLimit, upperLimit))
	{
		auto exactLower = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, lowerLimit);
		auto exactUpper = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, upperLimit);
		theValue = fmin(exactLower, exactUpper);
	}
	else if (EnteringHeightZone(theHeight, thePreviousHeight, lowerLimit))
	{
		double exactLower = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, lowerLimit);
		theValue = fmin(exactLower, theValue);
	}
	else if (LeavingHeightZone(theHeight, thePreviousHeight, upperLimit))
	{
		double exactUpper = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, upperLimit);
		theValue = fmin(exactUpper, theValue);
		itsOutOfBoundHeights[itsIndex] = true;
	}

	if (IsMissing(Value()) || theValue < Value())
	{
		Value(theValue);
	}
}

/* ----------------- */

void modifier_maxmin::Init(const std::vector<double>& theData, const std::vector<double>& theHeights)
{
	modifier::Init(theData, theHeights);
	itsMaximumResult.resize(theData.size(), MissingDouble());
}

const std::vector<double>& modifier_maxmin::Result() const
{
	itsResult.insert(itsResult.end(), itsMaximumResult.begin(), itsMaximumResult.end());
	return itsResult;
}

void modifier_maxmin::Calculate(double theValue, double theHeight, double thePreviousValue, double thePreviousHeight)
{
	double lowerLimit = itsLowerHeight[itsIndex];
	double upperLimit = itsUpperHeight[itsIndex];

	double bigger = theValue, smaller = theValue;

	if (BetweenLevels(theHeight, thePreviousHeight, lowerLimit, upperLimit))
	{
		auto exactLower = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, lowerLimit);
		auto exactUpper = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, upperLimit);

		smaller = fmin(exactLower, exactUpper);
		bigger = fmax(exactLower, exactUpper);
	}
	else if (EnteringHeightZone(theHeight, thePreviousHeight, lowerLimit))
	{
		double exactLower = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, lowerLimit);

		smaller = fmin(exactLower, theValue);
		bigger = fmax(exactLower, theValue);
	}
	else if (LeavingHeightZone(theHeight, thePreviousHeight, upperLimit))
	{
		double exactUpper = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, upperLimit);

		smaller = fmin(exactUpper, theValue);
		bigger = fmax(exactUpper, theValue);

		itsOutOfBoundHeights[itsIndex] = true;
	}

	if (IsMissing(Value()))
	{
		// Set min == max
		itsResult[itsIndex] = smaller;
		itsMaximumResult[itsIndex] = bigger;
	}
	else
	{
		itsMaximumResult[itsIndex] = fmax(bigger, itsMaximumResult[itsIndex]);
		itsResult[itsIndex] = fmin(smaller, itsResult[itsIndex]);
	}
}

/* ----------------- */

void modifier_sum::Calculate(double theValue, double theHeight, double thePreviousValue, double thePreviousHeight)
{
	if (IsMissing(Value()))  // First value
	{
		Value(theValue);
	}
	else
	{
		double val = Value();
		Value(theValue + val);
	}
}

/* ----------------- */

void modifier_mean::Init(const std::vector<double>& theData, const std::vector<double>& theHeights)
{
	if (itsResult.size() == 0)
	{
		modifier::Init(theData, theHeights);
		itsRange.resize(theData.size(), 0);
	}
}

void modifier_mean::Calculate(double theValue, double theHeight, double thePreviousValue, double thePreviousHeight)
{
	if (IsMissing(Value()))  // First value
	{
		Value(0);
	}

	double lowerLimit = itsLowerHeight[itsIndex];
	double upperLimit = itsUpperHeight[itsIndex];

	// check if averaging interval is larger then 0. Otherwise skip this gridpoint and return average value of 0.
	if (lowerLimit == upperLimit)
	{
		itsOutOfBoundHeights[itsIndex] = true;
		return;
	}

	double val = Value();

	if (BetweenLevels(theHeight, thePreviousHeight, lowerLimit, upperLimit))
	{
		auto lowerValue = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, lowerLimit);
		auto upperValue = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, upperLimit);

		Value((upperValue + lowerValue) / 2 * (upperLimit - lowerLimit));
		itsRange[itsIndex] += upperLimit - lowerLimit;
		// if upper height is passed for this grid point set OutOfBoundHeight = "true" to skip calculation of the
		// integral in following iterations
		itsOutOfBoundHeights[itsIndex] = true;
	}
	else if (EnteringHeightZone(theHeight, thePreviousHeight, lowerLimit))
	{
		double lowerValue = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, lowerLimit);
		Value((lowerValue + theValue) / 2 * (theHeight - lowerLimit) + val);
		itsRange[itsIndex] += theHeight - lowerLimit;
	}
	else if (LeavingHeightZone(theHeight, thePreviousHeight, upperLimit))
	{
		double upperValue = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, upperLimit);

		Value((upperValue + thePreviousValue) / 2 * (upperLimit - thePreviousHeight) + val);
		itsRange[itsIndex] += upperLimit - thePreviousHeight;
	}
	else if (!IsMissing(thePreviousHeight) && !IsMissing(thePreviousValue))
	{
		Value((thePreviousValue + theValue) / 2 * (theHeight - thePreviousHeight) + val);
		itsRange[itsIndex] += theHeight - thePreviousHeight;
	}
}

const std::vector<double>& modifier_mean::Result() const
{
	for (size_t i = 0; i < itsResult.size(); i++)
	{
		double val = itsResult[i];

		if (IsMissing(itsRange[i]))
		{
			itsResult[i] = MissingDouble();
		}
		else if (!IsMissing(val) && fabs(itsRange[i]) > 0.0)
		{
			itsResult[i] = val / itsRange[i];
		}
	}

	return itsResult;
}

/* ----------------- */

void modifier_count::Init(const std::vector<double>& theData, const std::vector<double>& theHeights)
{
	if (itsResult.size() == 0)
	{
		modifier::Init(theData, theHeights);
		std::fill(itsResult.begin(), itsResult.end(), 0.);
	}
}

void modifier_count::Calculate(double theValue, double theHeight, double thePreviousValue, double thePreviousHeight)
{
	assert(itsFindValue.size());
	double findValue = itsFindValue[itsIndex];

	// First level

	if (IsMissing(thePreviousValue))
	{
		return;
	}

	double lowerLimit = itsLowerHeight[itsIndex];
	double upperLimit = itsUpperHeight[itsIndex];

	if (BetweenLevels(theHeight, thePreviousHeight, lowerLimit, upperLimit))
	{
		auto exactLower = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, lowerLimit);
		auto exactUpper = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, upperLimit);

		if ((exactLower <= findValue && exactUpper >= findValue) ||
		    (exactLower >= findValue && exactUpper <= findValue))
		{
			Value(Value() + 1);
		}

		return;
	}
	else if (EnteringHeightZone(theHeight, thePreviousHeight, lowerLimit))
	{
		theValue = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, lowerLimit);
	}
	else if (LeavingHeightZone(theHeight, thePreviousHeight, upperLimit))
	{
		theValue = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, upperLimit);
		itsOutOfBoundHeights[itsIndex] = true;
	}

	if ((thePreviousValue <= findValue && theValue >= findValue)      // upward trend
	    || (thePreviousValue >= findValue && theValue <= findValue))  // downward trend
	{
		double val = Value();
		Value(val + 1);
	}
}

/* ----------------- */

void modifier_findheight::Clear(double fillValue)
{
	modifier::Clear(fillValue);
	std::fill(itsFoundNValues.begin(), itsFoundNValues.end(), 0);
	itsValuesFound = 0;
}

bool modifier_findheight::CalculationFinished() const
{
	return (itsResult.size() && (itsValuesFound == itsResult.size() ||
	                             static_cast<size_t>(count(itsOutOfBoundHeights.begin(), itsOutOfBoundHeights.end(),
	                                                       true)) == itsResult.size()));
}

void modifier_findheight::Init(const std::vector<double>& theData, const std::vector<double>& theHeights)
{
	if (itsResult.size() == 0)
	{
		assert(theData.size() == theHeights.size());
		assert(theData.size());

		modifier::Init(theData, theHeights);

		itsFoundNValues.resize(itsResult.size(), 0);

		// If Find values have missing values we can't process those grid points

		for (size_t i = 0; i < itsFindValue.size(); i++)
		{
			if (IsMissing(itsFindValue[i]))
			{
				itsOutOfBoundHeights[i] = true;
			}
		}

		itsValuesFound = 0;
	}
}

void modifier_findheight::Calculate(double theValue, double theHeight, double thePreviousValue,
                                    double thePreviousHeight)
{
	assert(itsFindValue.size() && itsIndex < itsFindValue.size());

	double findValue = itsFindValue[itsIndex];

	if (itsFindNthValue > 0 && !IsMissing(Value()))
	{
		return;
	}

	if (IsMissingValue(thePreviousValue))
	{
		return;
	}

	const double lowerLimit = itsLowerHeight[itsIndex];
	const double upperLimit = itsUpperHeight[itsIndex];

	if (EnteringHeightZone(theHeight, thePreviousHeight, lowerLimit))
	{
		thePreviousValue = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, lowerLimit);
		thePreviousHeight = lowerLimit;
	}
	else if (LeavingHeightZone(theHeight, thePreviousHeight, upperLimit))
	{
		theValue = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, upperLimit);
		theHeight = upperLimit;

		itsOutOfBoundHeights[itsIndex] = true;
	}

	if ((thePreviousValue <= findValue && theValue >= findValue) ||
	    (thePreviousValue > findValue && theValue <= findValue))
	{
		double actualHeight =
		    interpolation::Linear(findValue, thePreviousValue, theValue, thePreviousHeight, theHeight);

		if (!IsMissing(actualHeight))
		{
			assert(!itsHeightInMeters || (actualHeight >= lowerLimit && actualHeight <= upperLimit));
			assert(itsHeightInMeters || (actualHeight <= lowerLimit && actualHeight >= upperLimit));

			if (itsFindNthValue != 0)
			{
				itsFoundNValues[itsIndex] += 1;

				if (itsFindNthValue == itsFoundNValues[itsIndex])
				{
					Value(actualHeight);
					itsValuesFound++;
					itsOutOfBoundHeights[itsIndex] = true;
				}
			}
			else
			{
				// Search for the last value
				Value(actualHeight);
			}
		}
	}
}

/* ----------------- */

void modifier_findheight_gt::FindNth(size_t theNth)
{
	if (theNth > 1)
	{
		throw std::runtime_error("modifier_findheight_gt: For Nth values only 0 or 1 are accepted");
	}

	itsFindNthValue = theNth;
}

void modifier_findheight_gt::Calculate(double theValue, double theHeight, double thePreviousValue,
                                       double thePreviousHeight)
{
	assert(itsFindValue.size() && itsIndex < itsFindValue.size());
	const double findValue = itsFindValue[itsIndex];

	if (itsFindNthValue > 0 && !IsMissing(Value()))
	{
		return;
	}

	const double lowerLimit = itsLowerHeight[itsIndex];
	const double upperLimit = itsUpperHeight[itsIndex];

	// Check if we have just entered or just leaving a height zone
	if (EnteringHeightZone(theHeight, thePreviousHeight, lowerLimit))
	{
		thePreviousValue = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, lowerLimit);
		thePreviousHeight = lowerLimit;

		// Lower edge might be valid, check it before moving to check current height

		if (thePreviousValue > findValue)
		{
			itsFoundNValues[itsIndex] += 1;

			if (itsFindNthValue == itsFoundNValues[itsIndex])
			{
				Value(thePreviousHeight);
				itsValuesFound++;
				itsOutOfBoundHeights[itsIndex] = true;
			}
			else
			{
				Value(thePreviousHeight);
			}
		}
	}
	else if (LeavingHeightZone(theHeight, thePreviousHeight, upperLimit))
	{
		theValue = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, upperLimit);
		theHeight = upperLimit;
		itsOutOfBoundHeights[itsIndex] = true;
	}

	// Entering area
	if (theValue > findValue && (thePreviousValue < findValue || IsMissing(thePreviousValue)))
	{
		// if last value is searched, pick actual level value
		if (itsFindNthValue == 0)
		{
			Value(theHeight);
		}
		// else we need to interpolate earlier value
		else
		{
			if (!IsMissing(thePreviousValue) && !IsMissing(thePreviousHeight))
			{
				theHeight = interpolation::Linear(findValue, thePreviousValue, theValue, thePreviousHeight, theHeight);
			}

			itsFoundNValues[itsIndex] += 1;

			if (itsFindNthValue == itsFoundNValues[itsIndex])
			{
				Value(theHeight);
				itsValuesFound++;
				itsOutOfBoundHeights[itsIndex] = true;
			}
		}
	}
	// In area
	else if (theValue > findValue && thePreviousValue > findValue && itsFindNthValue == 0)
	{
		Value(theHeight);
	}
	// Leaving area
	else if (theValue < findValue && (!IsMissing(thePreviousValue) && thePreviousValue > findValue))
	{
		if (itsFindNthValue == 0)
		{
			Value(theHeight);
		}
		else
		{
			if (!IsMissing(thePreviousHeight))
			{
				theHeight = interpolation::Linear(findValue, thePreviousValue, theValue, thePreviousHeight, theHeight);
			}

			itsFoundNValues[itsIndex] += 1;

			if (itsFindNthValue == itsFoundNValues[itsIndex])
			{
				Value(theHeight);
				itsValuesFound++;
				itsOutOfBoundHeights[itsIndex] = true;
			}
		}
	}
}

/* ----------------- */

void modifier_findheight_lt::FindNth(size_t theNth)
{
	if (theNth != 0 && theNth != 1)
	{
		throw std::runtime_error("modifier_findheight_lt: For Nth values only 0 or 1 are accepted");
	}

	itsFindNthValue = theNth;
}

void modifier_findheight_lt::Calculate(double theValue, double theHeight, double thePreviousValue,
                                       double thePreviousHeight)
{
	assert(itsFindValue.size() && itsIndex < itsFindValue.size());
	const double findValue = itsFindValue[itsIndex];

	if (itsFindNthValue > 0 && !IsMissing(Value()))
	{
		return;
	}

	const double lowerLimit = itsLowerHeight[itsIndex];
	const double upperLimit = itsUpperHeight[itsIndex];

	if (EnteringHeightZone(theHeight, thePreviousHeight, lowerLimit))
	{
		thePreviousValue = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, lowerLimit);
		thePreviousHeight = lowerLimit;

		// Lower edge might be valid, check it before moving to check current height

		if (thePreviousValue < findValue)
		{
			itsFoundNValues[itsIndex] += 1;

			if (itsFindNthValue == itsFoundNValues[itsIndex])
			{
				Value(thePreviousHeight);
				itsValuesFound++;
				itsOutOfBoundHeights[itsIndex] = true;
			}
			else
			{
				Value(thePreviousHeight);
			}
		}
	}
	else if (LeavingHeightZone(theHeight, thePreviousHeight, upperLimit))
	{
		theValue = ExactEdgeValue(theHeight, theValue, thePreviousHeight, thePreviousValue, upperLimit);
		theHeight = upperLimit;
		itsOutOfBoundHeights[itsIndex] = true;
	}

	// Entering area
	if (theValue < findValue && thePreviousValue > findValue)
	{
		// if last value is searched, pick actual level value
		if (itsFindNthValue == 0)
		{
			Value(theHeight);
		}
		// else we need to interpolate earlier value
		else
		{
			if (!IsMissing(thePreviousValue) && !IsMissing(thePreviousHeight))
			{
				theHeight = interpolation::Linear(findValue, thePreviousValue, theValue, thePreviousHeight, theHeight);
			}

			itsFoundNValues[itsIndex] += 1;

			if (itsFindNthValue == itsFoundNValues[itsIndex])
			{
				Value(theHeight);
				itsValuesFound++;
				itsOutOfBoundHeights[itsIndex] = true;
			}
		}
	}
	// In area
	else if (theValue < findValue && thePreviousValue < findValue && itsFindNthValue == 0)
	{
		Value(theHeight);
	}
	// Leaving area
	else if (theValue > findValue && thePreviousValue < findValue)
	{
		if (itsFindNthValue == 0)
		{
			Value(theHeight);
		}
		else
		{
			if (!IsMissing(thePreviousValue) && !IsMissing(thePreviousHeight))
			{
				theHeight = interpolation::Linear(findValue, thePreviousValue, theValue, thePreviousHeight, theHeight);
			}

			itsFoundNValues[itsIndex] += 1;

			if (itsFindNthValue == itsFoundNValues[itsIndex])
			{
				Value(theHeight);
				itsValuesFound++;
				itsOutOfBoundHeights[itsIndex] = true;
			}
		}
	}
}

/* ----------------- */

void modifier_findvalue::Init(const std::vector<double>& theData, const std::vector<double>& theHeights)
{
	if (itsResult.size() == 0)
	{
		assert(theData.size() == theHeights.size());

		modifier::Init(theData, theHeights);

		assert(itsFindValue.size());

		double lowestHeight = DEFAULT_MAXIMUM;  // sic
		double highestHeight = DEFAULT_MINIMUM;

		for (size_t i = 0; i < itsFindValue.size(); i++)
		{
			double h = itsFindValue[i];

			if (IsMissing(h))
			{
				itsOutOfBoundHeights[i] = true;
				continue;
			}

			if (h > highestHeight)
			{
				highestHeight = h;
			}
			if (h < lowestHeight)
			{
				lowestHeight = h;
			}
		}

		if (!itsHeightInMeters)
		{
			double tmp = lowestHeight;
			lowestHeight = highestHeight;
			highestHeight = tmp;
		}
		// Give some threshold to lowest and highest heights

		if (itsHeightInMeters)
		{
			lowestHeight = fmax(0, lowestHeight - 500);  // meters
			itsLowerHeight.resize(itsResult.size(), lowestHeight);
			itsUpperHeight.resize(itsResult.size(), highestHeight + 500);
		}
		else
		{
			lowestHeight = lowestHeight + 50;  // hectopascals
			itsLowerHeight.resize(itsResult.size(), lowestHeight);
			itsUpperHeight.resize(itsResult.size(), highestHeight - 50);
		}

		itsValuesFound = 0;

#ifdef EXTRADEBUG
		DumpVector(itsFindValue);
		DumpVector(itsLowerHeight);
		DumpVector(itsUpperHeight);
		DumpVector(itsOutOfBoundHeights);
#endif
	}
}

bool modifier_findvalue::CalculationFinished() const
{
	return (itsResult.size() && (itsValuesFound == itsResult.size() ||
	                             static_cast<size_t>(count(itsOutOfBoundHeights.begin(), itsOutOfBoundHeights.end(),
	                                                       true)) == itsResult.size()));
}

void modifier_findvalue::Calculate(double theValue, double theHeight, double thePreviousValue, double thePreviousHeight)
{
	assert(itsFindValue.size() && itsIndex < itsFindValue.size());

	double findHeight = itsFindValue[itsIndex];

	if (itsGridsProcessed == 0 &&
	    ((itsHeightInMeters && findHeight < theHeight) || (!itsHeightInMeters && findHeight > theHeight)))
	{
		// It's possible that the height requested is below the lowest hybrid level, meaning
		// that we cannot interpolate the value. In this case clamp the value to the lowest
		// hybrid level.

		// Clamp threshold is set to 20 meters or hPa: if the difference between requested height
		// and lowest hybrid level is larger that this then clamping is not done and
		// MissingDouble() is the result

		double diff = fabs(theHeight - findHeight);

		if (diff < 20)
		{
			Value(theValue);
			itsValuesFound++;
		}

		itsOutOfBoundHeights[itsIndex] = true;

		// previous was missing but the level we want is above current height
		return;
	}

	if ((thePreviousHeight <= findHeight && theHeight >= findHeight)      // upward trend
	    || (thePreviousHeight >= findHeight && theHeight <= findHeight))  // downward trend
	{
		double actualValue =
		    interpolation::Linear(findHeight, thePreviousHeight, theHeight, thePreviousValue, theValue);

		if (!IsMissing(actualValue))
		{
			Value(actualValue);
			itsValuesFound++;
			itsOutOfBoundHeights[itsIndex] = true;
		}
	}
}

/* ----------------- */

void modifier_integral::Calculate(double theValue, double theHeight, double thePreviousValue, double thePreviousHeight)
{
	if (IsMissingValue(Value()))  // First value
	{
		Value(0);
	}

	double lowerHeight = itsLowerHeight[itsIndex];
	double upperHeight = itsUpperHeight[itsIndex];

	double previousValue = itsPreviousValue[itsIndex];
	double previousHeight = itsPreviousHeight[itsIndex];

	itsPreviousValue[itsIndex] = theValue;
	itsPreviousHeight[itsIndex] = theHeight;

	if (previousHeight < lowerHeight && theHeight > lowerHeight)
	{
		double val = Value();
		double lowerValue = interpolation::Linear(lowerHeight, previousHeight, theHeight, previousValue, theValue);
		Value((lowerValue + theValue) / 2 * (theHeight - lowerHeight) + val);
	}
	else if (previousHeight < upperHeight && theHeight > upperHeight)
	{
		double val = Value();
		double upperValue = interpolation::Linear(upperHeight, previousHeight, theHeight, previousValue, theValue);
		Value((upperValue + previousValue) / 2 * (upperHeight - previousHeight) + val);
	}
	else if (!IsMissing(previousHeight) && previousHeight >= lowerHeight && theHeight <= upperHeight)
	{
		double val = Value();
		Value((previousValue + theValue) / 2 * (theHeight - previousHeight) + val);
	}
}

/* ----------------- */

void modifier_plusminusarea::Init(const std::vector<double>& theData, const std::vector<double>& theHeights)
{
	if (itsResult.size() == 0)
	{
		modifier::Init(theData, theHeights);

		itsPlusArea.resize(theData.size(), 0);
		itsMinusArea.resize(theData.size(), 0);
	}
}

void modifier_plusminusarea::Calculate(double theValue, double theHeight, double thePreviousValue,
                                       double thePreviousHeight)
{
	double lowerHeight = itsLowerHeight[itsIndex];
	double upperHeight = itsUpperHeight[itsIndex];

	// check if interval is larger then 0. Otherwise skip this gridpoint and return value of 0.
	if (lowerHeight == upperHeight)
	{
		itsOutOfBoundHeights[itsIndex] = true;
		return;
	}

	// integrate numerically with separating positive from negative area under the curve.
	// find lower bound

	// TODO: add between levels case
	if (EnteringHeightZone(theHeight, thePreviousHeight, lowerHeight))
	{
		double lowerValue =
		    interpolation::Linear(lowerHeight, thePreviousHeight, theHeight, thePreviousValue, theValue);
		// zero is crossed from negative to positive: Interpolate height where zero is crossed and integrate positive
		// and negative area separately
		if (lowerValue < 0 && theValue > 0)
		{
			double zeroHeight = interpolation::Linear(0.0, lowerValue, theValue, lowerHeight, theHeight);
			itsMinusArea[itsIndex] += lowerValue / 2 * (zeroHeight - lowerHeight);
			itsPlusArea[itsIndex] += theValue / 2 * (theHeight - zeroHeight);
		}
		// zero is crossed from positive to negative
		else if (lowerValue > 0 && theValue < 0)
		{
			double zeroHeight = interpolation::Linear(0.0, lowerValue, theValue, lowerHeight, theHeight);

			itsPlusArea[itsIndex] += lowerValue / 2 * (zeroHeight - lowerHeight);
			itsMinusArea[itsIndex] += theValue / 2 * (theHeight - zeroHeight);
		}
		// whole interval is in the negative area
		else if (lowerValue <= 0 && theValue <= 0)
		{
			itsMinusArea[itsIndex] += (lowerValue + theValue) / 2 * (theHeight - lowerHeight);
		}
		// whole interval is in the positive area
		else
		{
			itsPlusArea[itsIndex] += (lowerValue + theValue) / 2 * (theHeight - lowerHeight);
		}
	}
	// find upper bound
	else if (LeavingHeightZone(theHeight, thePreviousHeight, upperHeight))
	{
		double upperValue =
		    interpolation::Linear(upperHeight, thePreviousHeight, theHeight, thePreviousValue, theValue);
		// zero is crossed from negative to positive
		if (thePreviousValue < 0 && upperValue > 0)
		{
			double zeroHeight =
			    interpolation::Linear(0.0, thePreviousValue, upperValue, thePreviousHeight, upperHeight);
			itsMinusArea[itsIndex] += thePreviousValue / 2 * (zeroHeight - thePreviousHeight);
			itsPlusArea[itsIndex] += upperValue / 2 * (upperHeight - zeroHeight);
		}
		// zero is crossed from positive to negative
		else if (thePreviousValue > 0 && upperValue < 0)
		{
			double zeroHeight =
			    interpolation::Linear(0.0, thePreviousValue, upperValue, thePreviousHeight, upperHeight);
			itsPlusArea[itsIndex] += thePreviousValue / 2 * (zeroHeight - thePreviousHeight);
			itsMinusArea[itsIndex] += upperValue / 2 * (upperHeight - zeroHeight);
		}
		// whole interval is in the negative area
		else if (thePreviousValue <= 0 && upperValue <= 0)
		{
			itsMinusArea[itsIndex] += (thePreviousValue + upperValue) / 2 * (upperHeight - thePreviousHeight);
		}
		// whole interval is in the positive area
		else
		{
			itsPlusArea[itsIndex] += (thePreviousValue + upperValue) / 2 * (upperHeight - thePreviousHeight);
		}
		// if upper height is passed for this grid point set OutOfBoundHeight = "true" to skip calculation of the
		// integral in following iterations
		itsOutOfBoundHeights[itsIndex] = true;
	}
	else if (!IsMissing(thePreviousHeight) && thePreviousHeight >= lowerHeight && theHeight <= upperHeight)
	{
		// zero is crossed from negative to positive
		if (thePreviousValue < 0 && theValue > 0)
		{
			double zeroHeight = interpolation::Linear(0.0, thePreviousValue, theValue, thePreviousHeight, theHeight);
			itsMinusArea[itsIndex] += thePreviousValue / 2 * (zeroHeight - thePreviousHeight);
			itsPlusArea[itsIndex] += theValue / 2 * (theHeight - zeroHeight);
		}
		// zero is crossed from positive to negative
		else if (thePreviousValue > 0 && theValue < 0)
		{
			double zeroHeight = interpolation::Linear(0.0, thePreviousValue, theValue, thePreviousHeight, theHeight);
			itsPlusArea[itsIndex] += thePreviousValue / 2 * (zeroHeight - thePreviousHeight);
			itsMinusArea[itsIndex] += theValue / 2 * (theHeight - zeroHeight);
		}
		// whole interval is in the negative area
		else if (thePreviousValue <= 0 && theValue <= 0)
		{
			itsMinusArea[itsIndex] += (thePreviousValue + theValue) / 2 * (theHeight - thePreviousHeight);
		}
		// whole interval is in the positive area
		else
		{
			itsPlusArea[itsIndex] += (thePreviousValue + theValue) / 2 * (theHeight - thePreviousHeight);
		}
	}
}

const std::vector<double>& modifier_plusminusarea::Result() const
{
	itsPlusArea.insert(itsPlusArea.end(), itsMinusArea.begin(),
	                   itsMinusArea.end());  // append MinusArea at the end of PlusArea
	return itsPlusArea;                      // return PlusMinusArea
}
