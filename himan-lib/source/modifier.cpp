/**
 * @file modifier.cpp
 * @author partio
 */

#include "modifier.h"

using namespace himan;

modifier::modifier()
	: itsMissingValuesAllowed(false)
	, itsFindNthValue(1) // first
{
}

const std::vector<double>& modifier::Result() const
{
	assert(itsResult.size());
	
	return itsResult;
}

bool modifier::CalculationFinished() const
{
		return false;
}

void modifier::Clear(double fillValue)
{
	std::fill(itsResult.begin(), itsResult.end(), fillValue);
}

bool modifier::IsMissingValue(double theValue) const
{
	if (theValue == kFloatMissing)
	{
		return true;
	}

	return false;
}

void modifier::FindValue(const std::vector<double>& theFindValue)
{
	itsFindValue = theFindValue;
}

size_t modifier::FindNth() const
{
	return itsFindNthValue;
}

void modifier::FindNth(size_t theNth)
{
	itsFindNthValue = theNth;
}

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

		itsResult.resize(theData.size(), kFloatMissing);

	}
}

void modifier::Process(const std::vector<double>& theData, const std::vector<double>& theHeights)
{

	Init(theData, theHeights);
	
	assert(itsResult.size() == theData.size() && itsResult.size() == theHeights.size());

	for (itsIndex = 0; itsIndex < itsResult.size(); itsIndex++)
	{
		Calculate(theData[itsIndex], theHeights[itsIndex]);
	}
}

std::ostream& modifier::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;

	file << "__itsMissingValuesAllowed__ " << itsMissingValuesAllowed << std::endl;
	file << "__itsFindValue__ size " << itsFindValue.size() << std::endl;
	file << "__itsFindNthValue__ " << itsFindNthValue << std::endl;
	file << "__itsResult__ size " << itsResult.size() << std::endl;
	file << "__itsIndex__ " << itsIndex << std::endl;


	return file;
}

/* ----------------- */

void modifier_max::Calculate(double theValue, double theHeight)
{

	if (IsMissingValue(theValue))
	{
		return;
	}

	if (IsMissingValue(Value()) || theValue > Value())
	{
		Value(theValue);
	}
}

/* ----------------- */

void modifier_min::Calculate(double theValue, double theHeight)
{
	if (IsMissingValue(theValue))
	{
		return;
	}

	if (IsMissingValue(Value()) || theValue < Value())
	{
		Value(theValue);
	}
}

/* ----------------- */

void modifier_maxmin::Init(const std::vector<double>& theData, const std::vector<double>& theHeights)
{
	if (itsResult.size() == 0)
	{
		assert(theData.size() == theHeights.size());

		itsResult.resize(theData.size(), kFloatMissing);
		itsMaximumResult.resize(theData.size(), kFloatMissing);
	}
}

const std::vector<double>& modifier_maxmin::Result() const
{
	itsResult.insert(itsResult.end(), itsMaximumResult.begin(), itsMaximumResult.end());
	return itsResult;
}

void modifier_maxmin::Calculate(double theValue, double theHeight)
{
	if (IsMissingValue(theValue))
	{
		return;
	}

	// Set min == max

	if (IsMissingValue(Value()))
	{
		itsResult[itsIndex] = theValue;
		itsMaximumResult[itsIndex] = theValue;
	}
	else
	{
		if (theValue > itsMaximumResult[itsIndex])
		{
			itsMaximumResult[itsIndex] = theValue;
		}

		if (theValue < itsResult[itsIndex])
		{
			itsResult[itsIndex] = theValue;
		}
	}
}

/* ----------------- */

void modifier_sum::Calculate(double theValue, double theHeight)
{
	if (IsMissingValue(theValue))
	{
		return;
	}

	if (IsMissingValue(Value())) // First value
	{
		Value(theValue);
	}
	else
	{
		double val = Value();
		Value(theValue+val);
	}
}

/* ----------------- */

void modifier_mean::Init(const std::vector<double>& theData, const std::vector<double>& theHeights)
{

	if (itsResult.size() == 0)
	{
		assert(theData.size() == theHeights.size());

		itsResult.resize(theData.size(), kFloatMissing);

		itsValuesCount.resize(itsResult.size(), 0);
	}
}

void modifier_mean::Calculate(double theValue, double theHeight)
{
	if (IsMissingValue(theValue))
	{
		return;
	}

	itsValuesCount[itsIndex] += 1;

	if (IsMissingValue(Value())) // First value
	{
		Value(theValue);
	}
	else
	{
		double val = Value();
		Value(val + theValue);
	}	
}

const std::vector<double>& modifier_mean::Result() const
{

	for (size_t i = 0; i < itsResult.size(); i++)
	{
		double val = itsResult[i];
		size_t count = itsValuesCount[i];

		if (!IsMissingValue(val) && count != 0)
		{
			itsResult[i] = val / static_cast<double> (count);
		}		
	}

	return itsResult;
}

/* ----------------- */

void modifier_count::Init(const std::vector<double>& theData, const std::vector<double>& theHeights)
{

	if (itsResult.size() == 0)
	{
		assert(theData.size() == theHeights.size());

		itsResult.resize(theData.size(), 0);

		itsPreviousValue.resize(itsResult.size(), kFloatMissing);
	}
}


void modifier_count::Calculate(double theValue, double theHeight)
{
	assert(itsFindValue.size());
	
	double findValue = itsFindValue[itsIndex];

	if (IsMissingValue(theValue) || IsMissingValue(findValue))
	{
		return;
	}

	double previousValue = itsPreviousValue[itsIndex];

	itsPreviousValue[itsIndex] = theValue;

	// First level

	if (IsMissingValue(previousValue))
	{
		return;
	}

	/**
	 *
	 * If lower value is found and current value is above wanted value, wanted value
	 * is found.
	 *
	 * Made up example
	 *
	 * How many times does value 11 exist inside a value range
	 *
	 * Input data set:
	 *
	 * Value
	 *
	 * 10
	 * --- Value 11 is found between these levels" --
	 * 12
	 *  9
	 *  9
	 * --- Value 11 is found between these levels! --
	 * 16	 
	 * 17
	 *
	 * The answer is: two times (as far as we know).
	 */
	
	if ((previousValue <= findValue && theValue >= findValue) // updward trend
			||
		(previousValue >= findValue && theValue <= findValue)) // downward trend
	{
		Value() == kFloatMissing ? Value(1) : Value(Value() + 1);
	}	
}

/* ----------------- */

void modifier_findheight::Clear(double fillValue)
{
	std::fill(itsResult.begin(), itsResult.end(), fillValue);
	std::fill(itsPreviousValue.begin(), itsPreviousValue.end(), fillValue);
	std::fill(itsPreviousHeight.begin(), itsPreviousHeight.end(), fillValue);
	std::fill(itsFoundNValues.begin(), itsFoundNValues.end(), 0);
}

bool modifier_findheight::CalculationFinished() const
{
	return itsValuesFound == itsResult.size();
}

void modifier_findheight::Init(const std::vector<double>& theData, const std::vector<double>& theHeights)
{
	if (itsResult.size() == 0)
	{
		assert(theData.size() == theHeights.size());
		assert(theData.size());

		itsResult.resize(theData.size(), kFloatMissing);
		itsPreviousValue.resize(itsResult.size(), kFloatMissing);
		itsPreviousHeight.resize(itsResult.size(), kFloatMissing);
		itsFoundNValues.resize(itsResult.size(), 0);

		itsValuesFound = 0;
	}
}

void modifier_findheight::Calculate(double theValue, double theHeight)
{

	assert(itsFindValue.size());
	
	double findValue = itsFindValue[itsIndex];
	
	if (IsMissingValue(theValue) || IsMissingValue(findValue) || (itsFindNthValue > 0 && !IsMissingValue(Value())))
	{
		return;
	}

	if (fabs(theValue - findValue) < 1e-5)
	{
		Value(theHeight);
		return;
	}

	double previousValue = itsPreviousValue[itsIndex];
	double previousHeight = itsPreviousHeight[itsIndex];

	itsPreviousValue[itsIndex] = theValue;
	itsPreviousHeight[itsIndex] = theHeight;

	if (IsMissingValue(previousValue))
	{
		return;
	}

	/**
	 *
	 * If lower value is found and current value is above wanted value, do the interpolation.
	 *
	 * Made up example
	 *
	 * Hight range: 120 - 125
	 * What is the height when parameter value is 15?
	 *
	 * Input data set:
	 *
	 * Height / Value
	 *
	 * 120 / 11
	 * 121 / 13
	 * 122 / 14
	 * --- Height of value 15 is found somewhere between these two levels! ---
	 * 123 / 16
	 * 124 / 19
	 * 125 / 19
	 *
	 * --> lowerValueThreshold = 14
	 * --> lowerHeightThreshold = 122
	 *
	 * --> theValue (== upperValueThreshold) = 16
	 * --> theHeight (== upperHeightThreshold) = 123
	 *
	 * Interpolate between (122,14),(123,16) to get the exact value !
	 * 
	 */

	if ((previousValue <= findValue && theValue >= findValue) || (previousValue >= findValue && theValue <= findValue))
	{
		double actualHeight = NFmiInterpolation::Linear(findValue, previousValue, theValue, previousHeight, theHeight);

		if (actualHeight != kFloatMissing)
		{
			if (itsFindNthValue != 0)
			{
				itsFoundNValues[itsIndex] += 1;

				if (itsFindNthValue == itsFoundNValues[itsIndex])
				{
					Value(actualHeight);
					itsValuesFound++;
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

void modifier_findvalue::Clear(double fillValue)
{
	std::fill(itsResult.begin(), itsResult.end(), fillValue);
	std::fill(itsPreviousValue.begin(), itsPreviousValue.end(), fillValue);
	std::fill(itsPreviousHeight.begin(), itsPreviousHeight.end(), fillValue);
}

void modifier_findvalue::Init(const std::vector<double>& theData, const std::vector<double>& theHeights)
{

	if (itsResult.size() == 0)
	{
		assert(theData.size() == theHeights.size());

		itsResult.resize(theData.size(), kFloatMissing);

		itsPreviousValue.resize(itsResult.size(), kFloatMissing);
		itsPreviousHeight.resize(itsResult.size(), kFloatMissing);

		itsValuesFound = 0;
	}
}

bool modifier_findvalue::CalculationFinished() const
{
	return itsValuesFound == itsResult.size();
}

void modifier_findvalue::Calculate(double theValue, double theHeight)
{

	double findHeight = itsFindValue[itsIndex];

	if (IsMissingValue(theValue) || !IsMissingValue(Value()) || IsMissingValue(findHeight))
	{
		return;
	}

	double previousValue = itsPreviousValue[itsIndex];
	double previousHeight = itsPreviousHeight[itsIndex];

	itsPreviousValue[itsIndex] = theValue;
	itsPreviousHeight[itsIndex] = theHeight;

	if (IsMissingValue(previousValue))
	{	
		return;
	}

	/**
	 *
	 * If lower height is found and current height is above wanted height,
	 * do the interpolation.
	 *
	 * Made up example
	 *
	 * Height: 124
	 * What is the parameter value?
	 *
	 * Input data set:
	 *
	 * Height / Value
	 *
	 * 120 / 11
	 * 121 / 13
	 * 122 / 14
	 * 123 / 16
	 * --- Value of height 124 is found somewhere between these two levels! ---
	 * 126 / 19
	 * 128 / 19
	 *
	 * --> lowerValueThreshold = 16
	 * --> lowerHeightThreshold = 123
	 *
	 * --> theValue (== upperValueThreshold) = 19
	 * --> theHeight (== upperHeightThreshold) = 126
	 *
	 * Interpolate between (123,16),(126,19) to get the exact value !
	 *
	 */

	if ((previousHeight <= findHeight && theHeight >= findHeight) // upward trend
			||
		(previousHeight >= findHeight && theHeight <= findHeight)) // downward trend
	{
		double actualValue = NFmiInterpolation::Linear(findHeight, previousHeight, theHeight, previousValue, theValue);

		if (actualValue != kFloatMissing)
		{
			Value(actualValue);
			itsValuesFound++;
		}
	}
}

