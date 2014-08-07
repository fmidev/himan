/**
 * @file modifier.cpp
 * @author partio
 */

#include "modifier.h"
#include "NFmiInterpolation.h"

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
	if (itsResult.size() > 0 && static_cast<size_t> (count(itsOutOfBoundHeights.begin(), itsOutOfBoundHeights.end(), true)) == itsResult.size())
	{
		return true;
	}
	
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

void modifier::LowerHeight(const std::vector<double>& theLowerHeight)
{
	itsLowerHeight = theLowerHeight;
#ifdef DEBUG
	double min = 1e38, max = -1e38, mean = 0;
	
	for (size_t i = 0; i < itsLowerHeight.size(); i++)
	{
		double val = itsLowerHeight[i];

		mean += val;
		if (val > max)
		{
			max = val;
		}
		else if (val < min)
		{
			min = val;
		}
	}

	std::cout << "itsLowerHeight min: " << min << " max: " << max << " mean: " << mean/static_cast<double> (itsLowerHeight.size()) << std::endl;
#endif
}

void modifier::UpperHeight(const std::vector<double>& theUpperHeight)
{
	itsUpperHeight = theUpperHeight;
#ifdef DEBUG
	double min = 1e38, max = -1e38, mean = 0;
	
	for (size_t i = 0; i < itsUpperHeight.size(); i++)
	{
		double val = itsUpperHeight[i];

		mean += val;
		if (val > max)
		{
			max = val;
		}
		else if (val < min)
		{
			min = val;
		}
	}

	std::cout << "itsUpperHeight min: " << min << " max: " << max << " mean: " << mean/static_cast<double> (itsLowerHeight.size()) << std::endl;
#endif
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
		itsOutOfBoundHeights.resize(theData.size(), false);
	}
}

bool modifier::Evaluate(double theValue, double theHeight)
{

	assert(itsIndex < itsOutOfBoundHeights.size());

	assert(theHeight != kFloatMissing);

	/*
	 * "upper" value relates to it being higher in the atmosphere
	 * meaning its value is higher.
	 *
	 * ie. lower limit 10, upper limit 100
	 *
	 * From this it follows that valid height is lower than upper limit and
	 * higher than lower limit
	 *
	 * TODO: If we'll ever be using pressure levels and Pa as unit this will
	 * need to be changed.
	*/

	// Absurd default limits if user has not specified any limits

	double upperLimit = 1e38;
	double lowerLimit = -1e38;

	if (!itsUpperHeight.empty())
	{
		upperLimit = itsUpperHeight[itsIndex];
	}

	if (!itsLowerHeight.empty())
	{
		lowerLimit = itsLowerHeight[itsIndex];
	}

	if (itsOutOfBoundHeights[itsIndex] == true)
	{
		return false;
	}
	else if (theHeight > upperLimit  || IsMissingValue(upperLimit) || IsMissingValue(lowerLimit))
	{
		// height is above given height range OR either level value is missing: stop processing of this grid point
		itsOutOfBoundHeights[itsIndex] = true;
		return false;
	}
	else if (theHeight < lowerLimit)
	{
		// height is below given height range, do not cancel calculation yet
		return false;
	}
	else if (IsMissingValue(theValue))
	{
		return false;
	}

	assert((lowerLimit == kFloatMissing || upperLimit == kFloatMissing) || (lowerLimit <= upperLimit));

	return true;
}

void modifier::Process(const std::vector<double>& theData, const std::vector<double>& theHeights)
{

	Init(theData, theHeights);
	
	assert(itsResult.size() == theData.size() && itsResult.size() == theHeights.size());

	for (itsIndex = 0; itsIndex < itsResult.size(); itsIndex++)
	{
		double theValue = theData[itsIndex], theHeight = theHeights[itsIndex];

		/*
		 * Evaluate() function is separated from Calculate() because Evaluate() is the
		 * same for all classes and therefore needs to defined only once
		 */
		
		if (!Evaluate(theValue, theHeight))
		{
			continue;
		}
		
		Calculate(theValue, theHeight);
	}
}

size_t modifier::HeightsCrossed() const
{
	return static_cast<size_t> (count(itsOutOfBoundHeights.begin(), itsOutOfBoundHeights.end(), true));
}

std::ostream& modifier::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;

	file << "__itsMissingValuesAllowed__ " << itsMissingValuesAllowed << std::endl;
	file << "__itsFindNthValue__ " << itsFindNthValue << std::endl;
	file << "__itsIndex__ " << itsIndex << std::endl;
	file << "__itsResult__ size " << itsResult.size() << std::endl;
	file << "__itsFindValue__ size " << itsFindValue.size() << std::endl;
	file << "__itsLowerHeight__ size " << itsLowerHeight.size() << std::endl;
	file << "__itsUpperHeight__ size " << itsUpperHeight.size() << std::endl;
	file << "__itsOutOfBoundHeights__ size " << itsOutOfBoundHeights.size() << std::endl;
	
	return file;
}

/* ----------------- */

void modifier_max::Calculate(double theValue, double theHeight)
{
	if (IsMissingValue(Value()) || theValue > Value())
	{
		Value(theValue);
	}
}

/* ----------------- */

void modifier_min::Calculate(double theValue, double theHeight)
{
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
		itsOutOfBoundHeights.resize(theData.size(), false);
	}
}

const std::vector<double>& modifier_maxmin::Result() const
{
	itsResult.insert(itsResult.end(), itsMaximumResult.begin(), itsMaximumResult.end());
	return itsResult;
}

void modifier_maxmin::Calculate(double theValue, double theHeight)
{
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

bool modifier_mean::Evaluate(double theValue, double theHeight)
{

	assert(itsIndex < itsOutOfBoundHeights.size());

	assert(theHeight != kFloatMissing);

	/*
	 * "upper" value relates to it being higher in the atmosphere
	 * meaning its value is higher.
	 *
	 * ie. lower limit 10, upper limit 100
	 *
	 * From this it follows that valid height is lower than upper limit and
	 * higher than lower limit
	 *
	 * TODO: If we'll ever be using pressure levels and Pa as unit this will
	 * need to be changed.
	 */

	/*
 	 * upper/lower limit check moved from evaluate function to calculate for the averaging case
	 */

	double upperLimit = 1e38;
	double lowerLimit = -1e38;

	if (!itsUpperHeight.empty())
	{
		upperLimit = itsUpperHeight[itsIndex];
	}

	if (!itsLowerHeight.empty())
	{
		lowerLimit = itsLowerHeight[itsIndex];
	}

	if (itsOutOfBoundHeights[itsIndex] == true)
	{
		return false;
	}
	else if (IsMissingValue(upperLimit) || IsMissingValue(lowerLimit))
	{
		// height is above given height range OR either level value is missing: stop processing of this grid point
		itsOutOfBoundHeights[itsIndex] = true;
		return false;
	}
	else if (IsMissingValue(theValue))
	{
		return false;
	}

	assert((lowerLimit == kFloatMissing || upperLimit == kFloatMissing) || (lowerLimit <= upperLimit));

	return true;
}

void modifier_mean::Init(const std::vector<double>& theData, const std::vector<double>& theHeights)
{

	if (itsResult.size() == 0)
	{
		assert(theData.size() == theHeights.size());

		itsResult.resize(theData.size(), kFloatMissing);

		itsOutOfBoundHeights.resize(itsResult.size(), false);

	}
}

/*
 *  The method used here to calculate the vertical average is limited to calculate average values for grids with constant vertical grid spacing only.
 *  Hybrid grids become denser close to the surface. This requires a more general method of calculation, i.e. the mean of a function (http://en.wikipedia.org/wiki/Mean_of_a_function).
 *  In this function that would mean in the most simple case to replace: 
 *  Value(val + theValue) -> Value(val + theValue*layer_depth)
 *  itsResult[i] = val / static_cast<double> (count) -> itsResult[i] = val / (itsUpperHeight - itsLowerHeight)
 *  But probably it's better to create a seperate modifier_integral class and let the average function call it and devide the result by (itsUpperHeight - itsLowerHeight).
 */

void modifier_mean::Calculate(double theValue, double theHeight)
{
	if (IsMissingValue(Value())) // First value
	{		
		Value(0);
	}

	double lowerHeight = itsLowerHeight[itsIndex];
	double upperHeight = itsUpperHeight[itsIndex];

	double previousValue = itsPreviousValue[itsIndex];
	double previousHeight = itsPreviousHeight[itsIndex];

	itsPreviousValue[itsIndex] = theValue;
	itsPreviousHeight[itsIndex] = theHeight;


	if (previousHeight <= lowerHeight && theHeight >= lowerHeight)
	{
		double val = Value();
		double lowerValue = NFmiInterpolation::Linear(lowerHeight, previousHeight, theHeight, previousValue, theValue);
		Value((lowerValue + theValue) / 2 * (theHeight - lowerHeight) + val);
	}
	else if (previousHeight <= upperHeight && theHeight >= upperHeight)
	{
		double val = Value();
		double upperValue = NFmiInterpolation::Linear(upperHeight, previousHeight, theHeight, previousValue, theValue);
		Value((upperValue + previousValue) / 2 * (upperHeight - previousHeight) + val);
	}
	else if (theHeight > lowerHeight && theHeight < upperHeight)
	{
		double val = Value();
		Value((previousValue + theValue) / 2 * (theHeight - previousHeight) + val);
	}
}
const std::vector<double>& modifier_mean::Result() const
{
	for (size_t i = 0; i < itsResult.size(); i++)
	{
		double val = itsResult[i];

		if (!IsMissingValue(val))
		{
			itsResult[i] = val / (itsUpperHeight[i] - itsLowerHeight[i]); 
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

		itsOutOfBoundHeights.resize(itsResult.size(), false);

	}
}


void modifier_count::Calculate(double theValue, double theHeight)
{
	assert(itsFindValue.size());
	
	double findValue = itsFindValue[itsIndex];

	if (IsMissingValue(findValue))
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
		double val = Value();
		Value() == kFloatMissing ? Value(1) : Value(val + 1);
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
	return (itsResult.size() && (itsValuesFound == itsResult.size() || static_cast<size_t> (count(itsOutOfBoundHeights.begin(), itsOutOfBoundHeights.end(), true)) == itsResult.size()));
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
		itsOutOfBoundHeights.resize(itsResult.size(), false);

		itsValuesFound = 0;
	}
}

void modifier_findheight::Calculate(double theValue, double theHeight)
{

	assert(itsFindValue.size() && itsIndex < itsFindValue.size());

	double findValue = itsFindValue[itsIndex];
	
	if (IsMissingValue(findValue) || (itsFindNthValue > 0 && !IsMissingValue(Value())))
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

		itsOutOfBoundHeights.resize(itsResult.size(), false);

		// Fake lower && upper heights

		double lowestHeight = 1e38;
		double highestHeight = -1;

		assert(itsFindValue.size());
		
		for (size_t i = 0; i < itsFindValue.size(); i++)
		{
			double h = itsFindValue[i];

			if (h == kFloatMissing)
			{
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

		itsLowerHeight.resize(itsResult.size(), lowestHeight);
		itsUpperHeight.resize(itsResult.size(), highestHeight);

		itsValuesFound = 0;
	}
}

bool modifier_findvalue::CalculationFinished() const
{
	return (itsResult.size() && (itsValuesFound == itsResult.size() || static_cast<size_t> (count(itsOutOfBoundHeights.begin(), itsOutOfBoundHeights.end(), true)) == itsResult.size()));
}

void modifier_findvalue::Calculate(double theValue, double theHeight)
{

	assert(itsFindValue.size() && itsIndex < itsFindValue.size());
	
	double findHeight = itsFindValue[itsIndex];

	if (!IsMissingValue(Value()) || IsMissingValue(findHeight))
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

/* ----------------- */

bool modifier_integral::Evaluate(double theValue, double theHeight)
{

	assert(itsIndex < itsOutOfBoundHeights.size());

	assert(theHeight != kFloatMissing);

	/*
	 * "upper" value relates to it being higher in the atmosphere
	 * meaning its value is higher.
	 *
	 * ie. lower limit 10, upper limit 100
	 *
	 * From this it follows that valid height is lower than upper limit and
	 * higher than lower limit
	 *
	 * TODO: If we'll ever be using pressure levels and Pa as unit this will
	 * need to be changed.
	 */

	/*
 	 * upper/lower limit check moved from evaluate function to calculate for the integration case
	 */

	double upperLimit = 1e38;
	double lowerLimit = -1e38;

	if (!itsUpperHeight.empty())
	{
		upperLimit = itsUpperHeight[itsIndex];
	}

	if (!itsLowerHeight.empty())
	{
		lowerLimit = itsLowerHeight[itsIndex];
	}

	if (itsOutOfBoundHeights[itsIndex] == true)
	{
		return false;
	}
	else if (IsMissingValue(upperLimit) || IsMissingValue(lowerLimit))
	{
		// height is above given height range OR either level value is missing: stop processing of this grid point
		itsOutOfBoundHeights[itsIndex] = true;
		return false;
	}
	else if (IsMissingValue(theValue))
	{
		return false;
	}

	assert((lowerLimit == kFloatMissing || upperLimit == kFloatMissing) || (lowerLimit <= upperLimit));

	return true;
}

void modifier_integral::Calculate(double theValue, double theHeight)
{
	if (IsMissingValue(Value())) // First value
	{		
		Value(0);
	}

	double lowerHeight = itsLowerHeight[itsIndex];
	double upperHeight = itsUpperHeight[itsIndex];

	double previousValue = itsPreviousValue[itsIndex];
	double previousHeight = itsPreviousHeight[itsIndex];

	itsPreviousValue[itsIndex] = theValue;
	itsPreviousHeight[itsIndex] = theHeight;


	if (previousHeight <= lowerHeight && theHeight >= lowerHeight)
	{
		double val = Value();
		double lowerValue = NFmiInterpolation::Linear(lowerHeight, previousHeight, theHeight, previousValue, theValue);
		Value((lowerValue + theValue) / 2 * (theHeight - lowerHeight) + val);
	}
	else if (previousHeight <= upperHeight && theHeight >= upperHeight)
	{
		double val = Value();
		double upperValue = NFmiInterpolation::Linear(upperHeight, previousHeight, theHeight, previousValue, theValue);
		Value((upperValue + previousValue) / 2 * (upperHeight - previousHeight) + val);
	}
	else if (theHeight > lowerHeight && theHeight < upperHeight)
	{
		double val = Value();
		Value((previousValue + theValue) / 2 * (theHeight - previousHeight) + val);
	}
}

