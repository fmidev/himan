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
	, itsModifierType(kUnknownModifierType)
	, itsHeightInMeters(true)
{
}

modifier::modifier(HPModifierType theModifierType)
	: itsMissingValuesAllowed(false)
	, itsFindNthValue(1) // first
	, itsModifierType(theModifierType)
	, itsHeightInMeters(true)
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
	size_t count = 0;
	
	for (size_t i = 0; i < itsLowerHeight.size(); i++)
	{
		double val = itsLowerHeight[i];

		if (IsMissingValue(val))
		{
			continue;
		}

		count++;
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

	if (count == 0)
	{
		std::cout << "itsUpperHeight all values are missing" << std::endl;
	}
	else
	{
		std::cout << "itsLowerHeight min: " << min << " max: " << max << " mean: " << mean/static_cast<double> (count) << std::endl;
	}
#endif
}

void modifier::UpperHeight(const std::vector<double>& theUpperHeight)
{
	itsUpperHeight = theUpperHeight;
#ifdef DEBUG
	double min = 1e38, max = -1e38, mean = 0;
	size_t count = 0;

	for (size_t i = 0; i < itsUpperHeight.size(); i++)
	{
		double val = itsUpperHeight[i];

		if (IsMissingValue(val))
		{
			continue;
		}
		
		count++;
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

	if (count == 0)
	{
		std::cout << "itsUpperHeight all values are missing" << std::endl;
	}
	else
	{
		std::cout << "itsUpperHeight min: " << min << " max: " << max << " mean: " << mean/static_cast<double> (count) << std::endl;
	}
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

	if (IsMissingValue(theHeight))
	{
		return false;
	}

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
	else if (itsHeightInMeters)
	{
		if (theHeight > upperLimit  || IsMissingValue(upperLimit) || IsMissingValue(lowerLimit))
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
	}
	else if (!itsHeightInMeters)
	{
		if (theHeight < upperLimit  || IsMissingValue(upperLimit) || IsMissingValue(lowerLimit))
		{
			itsOutOfBoundHeights[itsIndex] = true;
			return false;
		}
		else if (theHeight > lowerLimit)
		{
			// height is below given height range, do not cancel calculation yet
			return false;
		}
	}	
	else if (IsMissingValue(theValue))
	{
		return false;
	}

	assert((lowerLimit == kFloatMissing || upperLimit == kFloatMissing) || ((itsHeightInMeters && lowerLimit <= upperLimit) || (!itsHeightInMeters && lowerLimit >= upperLimit)));

	return true;
}

void modifier::Process(const std::vector<double>& theData, const std::vector<double>& theHeights)
{

	Init(theData, theHeights);
	
	//assert(itsResult.size() == theData.size() && itsResult.size() == theHeights.size());

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

HPModifierType modifier::Type() const
{
	return itsModifierType;
}

bool modifier::HeightInMeters() const
{
	return itsHeightInMeters;
}

void modifier::HeightInMeters(bool theHeightInMeters)
{
	itsHeightInMeters = theHeightInMeters;
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
	if (IsMissingValue(theValue))
	{
		return;
	}

	if (IsMissingValue(Value()))
	{
		// Set min == max
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

bool modifier_mean::Evaluate(double theValue, double theHeight)
{

	assert(itsIndex < itsOutOfBoundHeights.size());

	if (IsMissingValue(theHeight))
	{
		return false;
	}

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
	else if (itsOutOfBoundHeights[itsIndex])
	{
		// check if upper height for that grid point has been passed in the previous iteration
		return false;
	}
	else if (IsMissingValue(theValue))
	{
		return false;
	}

//	assert((lowerLimit == kFloatMissing || upperLimit == kFloatMissing) || (lowerLimit <= upperLimit));

	return true;
}

void modifier_mean::Init(const std::vector<double>& theData, const std::vector<double>& theHeights)
{

	if (itsResult.size() == 0)
	{
		assert(theData.size() == theHeights.size());

		itsResult.resize(theData.size(), kFloatMissing);
		itsRange.resize(theData.size(),0);
		itsPreviousValue.resize(itsResult.size(), kFloatMissing);
		itsPreviousHeight.resize(itsResult.size(), kFloatMissing);
	
		itsOutOfBoundHeights.resize(itsResult.size(), false);
	}
}

void modifier_mean::Calculate(double theValue, double theHeight)
{
	if (IsMissingValue(Value())) // First value
	{		
		Value(0);
	}

	double lowerHeight = -1e38;

	if (!itsLowerHeight.empty())
	{
		lowerHeight=itsLowerHeight[itsIndex];
	}

	double upperHeight = 1e38;

	if (!itsUpperHeight.empty())
	{
		upperHeight=itsUpperHeight[itsIndex];
	}

	double previousValue = itsPreviousValue[itsIndex];
	double previousHeight = itsPreviousHeight[itsIndex];

	itsPreviousValue[itsIndex] = theValue;
	itsPreviousHeight[itsIndex] = theHeight;
	
	// check if averaging interval is larger then 0. Otherwise skip this gridpoint and return average value of 0.
	if (lowerHeight == upperHeight)
	{
		itsOutOfBoundHeights[itsIndex] = true;
	}
	else if (previousHeight < lowerHeight && theHeight > lowerHeight)
	{
		double val = Value();
		double lowerValue = NFmiInterpolation::Linear(lowerHeight, previousHeight, theHeight, previousValue, theValue);
		Value((lowerValue + theValue) / 2 * (theHeight - lowerHeight) + val);
		itsRange[itsIndex] += theHeight - lowerHeight;
	}
	else if (previousHeight < upperHeight && theHeight > upperHeight)
	{
		double val = Value();
		double upperValue = NFmiInterpolation::Linear(upperHeight, previousHeight, theHeight, previousValue, theValue);
		Value((upperValue + previousValue) / 2 * (upperHeight - previousHeight) + val);
                itsRange[itsIndex] += upperHeight - previousHeight;
		// if upper height is passed for this grid point set OutOfBoundHeight = "true" to skip calculation of the integral in following iterations
		itsOutOfBoundHeights[itsIndex] = true;

	}
	else if (!(previousHeight == kFloatMissing) && previousHeight >= lowerHeight && theHeight <= upperHeight)
	{
		double val = Value();
		Value((previousValue + theValue) / 2 * (theHeight - previousHeight) + val);
                itsRange[itsIndex] += theHeight - previousHeight;
	}

}

const std::vector<double>& modifier_mean::Result() const
{
	for (size_t i = 0; i < itsResult.size(); i++)
	{
	
	double val = itsResult[i];

		if (!IsMissingValue(val))
		{
			itsResult[i] = val / itsRange[i]; 
		}
	}

	return itsResult;
}

bool modifier_mean::CalculationFinished() const
{
	if (itsResult.size() > 0 && static_cast<size_t> (count(itsOutOfBoundHeights.begin(), itsOutOfBoundHeights.end(), true)) == itsResult.size())
	{
		return true;
	}
	
	if (itsPreviousHeight > itsUpperHeight)
	{
		return true;
	}

	return false;

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
	std::fill(itsOutOfBoundHeights.begin(), itsOutOfBoundHeights.end(), false);
	itsValuesFound = 0;
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

	if (IsMissingValue(theHeight) || IsMissingValue(theValue))
	{
		return;
	}
	
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

	if ((previousValue <= findValue && theValue >= findValue) || (previousValue > findValue && theValue <= findValue))
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

void modifier_findvalue::Clear(double fillValue)
{
	std::fill(itsResult.begin(), itsResult.end(), fillValue);
	std::fill(itsPreviousValue.begin(), itsPreviousValue.end(), fillValue);
	std::fill(itsPreviousHeight.begin(), itsPreviousHeight.end(), fillValue);
	std::fill(itsOutOfBoundHeights.begin(), itsOutOfBoundHeights.end(), false);
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

		// Give some threshold to lowest and highest heights
		
		if (itsHeightInMeters)
		{
			lowestHeight = fmax(0, lowestHeight-500); // meters
			itsLowerHeight.resize(itsResult.size(), lowestHeight);
			itsUpperHeight.resize(itsResult.size(), highestHeight+500);
		}
		else
		{
			lowestHeight = lowestHeight+200; // hectopascals
			itsLowerHeight.resize(itsResult.size(), lowestHeight);
			itsUpperHeight.resize(itsResult.size(), highestHeight-200);
			
		}
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

	if (IsMissingValue(theHeight) || IsMissingValue(theValue))
	{
		return;
	}
	
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
		// It's possible that the height requested is below the lowest hybrid level, meaning
		// that we cannot interpolate the value. In this case clamp the value to the lowest
		// hybrid level.
		
		// Clamp threshold is set to 30 meters: if the difference between requested height
		// and lowest hybrid level is larger that this then clamping is not done and
		// kFloatMissing is the result

		double diff = fabs(theHeight - findHeight);
		if (findHeight < theHeight)
		{
			if (diff < 30)
			{
				Value(theValue);
				itsValuesFound++;
				itsOutOfBoundHeights[itsIndex] = true;
			}
		}

		// previous was missing but the level we want is above current height
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
			itsOutOfBoundHeights[itsIndex] = true;
		}		
	}
}

/* ----------------- */

void modifier_integral::Init(const std::vector<double>& theData, const std::vector<double>& theHeights)
{

	if (itsResult.size() == 0)
	{
		assert(theData.size() == theHeights.size());

		itsResult.resize(theData.size(), kFloatMissing);
		itsPreviousValue.resize(itsResult.size(), kFloatMissing);
		itsPreviousHeight.resize(itsResult.size(), kFloatMissing);
	
		itsOutOfBoundHeights.resize(itsResult.size(), false);

	}
}

bool modifier_integral::Evaluate(double theValue, double theHeight)
{

	assert(itsIndex < itsOutOfBoundHeights.size());

	if (IsMissingValue(theHeight))
	{
		return false;
	}

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

	assert((lowerLimit == kFloatMissing || upperLimit == kFloatMissing) || ((itsHeightInMeters && lowerLimit <= upperLimit) || (!itsHeightInMeters && lowerLimit >= upperLimit)));

	return true;
}

void modifier_integral::Calculate(double theValue, double theHeight)
{
	if (IsMissingValue(Value())) // First value
	{		
		Value(0);
	}

	double lowerHeight = -1e38;

	if (!itsLowerHeight.empty())
	{
		lowerHeight=itsLowerHeight[itsIndex];
	}

	double upperHeight = 1e38;

	if (!itsUpperHeight.empty())
	{
		upperHeight=itsUpperHeight[itsIndex];
	}

	double previousValue = itsPreviousValue[itsIndex];
	double previousHeight = itsPreviousHeight[itsIndex];

	itsPreviousValue[itsIndex] = theValue;
	itsPreviousHeight[itsIndex] = theHeight;


	if (previousHeight < lowerHeight && theHeight > lowerHeight)
	{
		double val = Value();
		double lowerValue = NFmiInterpolation::Linear(lowerHeight, previousHeight, theHeight, previousValue, theValue);
		Value((lowerValue + theValue) / 2 * (theHeight - lowerHeight) + val);
	}
	else if (previousHeight < upperHeight && theHeight > upperHeight)
	{
		double val = Value();
		double upperValue = NFmiInterpolation::Linear(upperHeight, previousHeight, theHeight, previousValue, theValue);
		Value((upperValue + previousValue) / 2 * (upperHeight - previousHeight) + val);
	}
	else if (!(previousHeight == kFloatMissing) && previousHeight >= lowerHeight && theHeight <= upperHeight)
	{
		double val = Value();
		Value((previousValue + theValue) / 2 * (theHeight - previousHeight) + val);
	}
}

bool modifier_integral::CalculationFinished() const
{
	if (itsResult.size() > 0 && static_cast<size_t> (count(itsOutOfBoundHeights.begin(), itsOutOfBoundHeights.end(), true)) == itsResult.size())
	{
		return true;
	}
	
	if (itsPreviousHeight > itsUpperHeight)
	{
		return true;
	}

	return false;

}

/* ----------------- */

bool modifier_plusminusarea::Evaluate(double theValue, double theHeight)
{
	assert(itsIndex < itsOutOfBoundHeights.size());
    if (IsMissingValue(theHeight) || IsMissingValue(theValue))
	{
		return false;
	}

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
	else if (itsOutOfBoundHeights[itsIndex])
	{
		// check if upper height for that grid point has been passed in the previous iteration
		return false;
	}

//	assert((lowerLimit == kFloatMissing || upperLimit == kFloatMissing) || (lowerLimit <= upperLimit));

	return true;
}

void modifier_plusminusarea::Init(const std::vector<double>& theData, const std::vector<double>& theHeights)
{

	if (itsResult.size() == 0)
	{
		assert(theData.size() == theHeights.size());

		itsPlusArea.resize(theData.size(), 0);
		itsMinusArea.resize(theData.size(), 0);
		itsPreviousValue.resize(theData.size(), kFloatMissing);
		itsPreviousHeight.resize(theData.size(), kFloatMissing);
	
		itsOutOfBoundHeights.resize(theData.size(), false);
	}
}

void modifier_plusminusarea::Process(const std::vector<double>& theData, const std::vector<double>& theHeights)
{

    Init(theData, theHeights);

	assert(itsPlusArea.size() == theData.size() && itsPlusArea.size() == theHeights.size());
	
	for (itsIndex = 0; itsIndex < itsPlusArea.size(); itsIndex++)
	{
		double theValue = theData[itsIndex], theHeight = theHeights[itsIndex];
		if (!Evaluate(theValue, theHeight))
		{
			continue;
		}
		
	    Calculate(theValue, theHeight);
	}
}

void modifier_plusminusarea::Calculate(double theValue, double theHeight)
{
	double lowerHeight = -1e38;

	if (!itsLowerHeight.empty())
	{
		lowerHeight=itsLowerHeight[itsIndex];
	}

	double upperHeight = 1e38;

	if (!itsUpperHeight.empty())
	{
		upperHeight=itsUpperHeight[itsIndex];
	}

	double previousValue = itsPreviousValue[itsIndex];
	double previousHeight = itsPreviousHeight[itsIndex];

	itsPreviousValue[itsIndex] = theValue;
	itsPreviousHeight[itsIndex] = theHeight;

	// check if interval is larger then 0. Otherwise skip this gridpoint and return value of 0.
	if (lowerHeight == upperHeight)
	{
		itsOutOfBoundHeights[itsIndex] = true;
	}
	// integrate numerically with separating positive from negative area under the curve.
	// find lower bound
	else if (previousHeight < lowerHeight && theHeight > lowerHeight)
	{
		double lowerValue = NFmiInterpolation::Linear(lowerHeight, previousHeight, theHeight, previousValue, theValue);
		// zero is crossed from negative to positive: Interpolate height where zero is crossed and integrate positive and negative area separately
		if (lowerValue < 0 && theValue > 0)
		{
			double zeroHeight = NFmiInterpolation::Linear(0.0, lowerValue, theValue, lowerHeight, theHeight);
			itsMinusArea[itsIndex] += lowerValue / 2 * (zeroHeight - lowerHeight);
			itsPlusArea[itsIndex] += theValue / 2 * (theHeight - zeroHeight);
		}
		// zero is crossed from positive to negative
		else if (lowerValue > 0 && theValue < 0)
		{
			double zeroHeight = NFmiInterpolation::Linear(0.0, lowerValue, theValue, lowerHeight, theHeight);
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
	else if (previousHeight < upperHeight && theHeight > upperHeight)
	{
		double upperValue = NFmiInterpolation::Linear(upperHeight, previousHeight, theHeight, previousValue, theValue);
		// zero is crossed from negative to positive
		if (previousValue < 0 && upperValue > 0)
		{
			double zeroHeight = NFmiInterpolation::Linear(0.0, previousValue, upperValue, previousHeight, upperHeight);
			itsMinusArea[itsIndex] += previousValue / 2 * (zeroHeight - previousHeight);
			itsPlusArea[itsIndex] += upperValue / 2 * (upperHeight - zeroHeight);
		}
		// zero is crossed from positive to negative
        else if (previousValue > 0 && upperValue < 0)
		{
			double zeroHeight = NFmiInterpolation::Linear(0.0, previousValue, upperValue, previousHeight, upperHeight);
			itsPlusArea[itsIndex] += previousValue / 2 * (zeroHeight - previousHeight);
			itsMinusArea[itsIndex] += upperValue / 2 * (upperHeight - zeroHeight);
		}
		// whole interval is in the negative area
		else if (previousValue <= 0 && upperValue <= 0)
		{
			itsMinusArea[itsIndex] += (previousValue + upperValue) / 2 * (upperHeight - previousHeight);
		}
		// whole interval is in the positive area
		else
		{
			itsPlusArea[itsIndex] += (previousValue + upperValue) / 2 * (upperHeight - previousHeight);
		}
		// if upper height is passed for this grid point set OutOfBoundHeight = "true" to skip calculation of the integral in following iterations
		itsOutOfBoundHeights[itsIndex] = true;
	}
	else if (!(previousHeight == kFloatMissing) && previousHeight >= lowerHeight && theHeight <= upperHeight)
	{
		// zero is crossed from negative to positive
		if (previousValue < 0 && theValue > 0)
		{
			double zeroHeight = NFmiInterpolation::Linear(0.0, previousValue, theValue, previousHeight, theHeight);
			itsMinusArea[itsIndex] += previousValue / 2 * (zeroHeight - previousHeight);
			itsPlusArea[itsIndex] += theValue / 2 * (theHeight - zeroHeight);
		}
		// zero is crossed from positive to negative
		else if (previousValue > 0 && theValue < 0)
		{
			double zeroHeight = NFmiInterpolation::Linear(0.0, previousValue, theValue, previousHeight, theHeight);
			itsPlusArea[itsIndex] += previousValue / 2 * (zeroHeight - previousHeight);
			itsMinusArea[itsIndex] += theValue / 2 * (theHeight - zeroHeight);
		}
		// whole interval is in the negative area
		else if (previousValue <= 0 && theValue <= 0)
		{
			itsMinusArea[itsIndex] += (previousValue + theValue) / 2 * (theHeight - previousHeight);
		}
		// whole interval is in the positive area
		else
		{
			itsPlusArea[itsIndex] += (previousValue + theValue) / 2 * (theHeight - previousHeight);
		}
	}
}

bool modifier_plusminusarea::CalculationFinished() const
{
	if (itsMinusArea.size() > 0 && static_cast<size_t> (count(itsOutOfBoundHeights.begin(), itsOutOfBoundHeights.end(), true)) == itsMinusArea.size())
	{
		return true;
	}
	
	if (itsPreviousHeight > itsUpperHeight)
	{
		return true;
	}

	return false;
}

const std::vector<double>& modifier_plusminusarea::Result() const
{
	itsPlusArea.insert(itsPlusArea.end(), itsMinusArea.begin(), itsMinusArea.end()); //append MinusArea at the end of PlusArea 
	return itsPlusArea; // return PlusMinusArea
}
