/**
 * @file modifier.cpp
 * @author partio
 */

#include "modifier.h"

using namespace himan;

modifier::modifier()
	: itsReturnHeight(true)
	, itsMissingValuesAllowed(false)
	, itsFindNthValue(1) // first
{
}

void modifier::Init(std::shared_ptr<const info> sourceInfo)
{
	itsResult = std::shared_ptr<info> (new info(*sourceInfo));
	itsResult->Create();
	
	Clear();

	itsResult->First();
}

std::shared_ptr<info> modifier::Results() const
{
	return itsResult;
}

bool modifier::NextLocation()
{
	return itsResult->NextLocation();
}

void modifier::ResetLocation()
{
	itsResult->ResetLocation();
}

size_t modifier::LocationIndex() const
{
	return itsResult->LocationIndex();
}

bool modifier::ReturnHeight() const
{
	return itsReturnHeight;
}

void modifier::ReturnHeight(bool theReturnHeight)
{
	itsReturnHeight = theReturnHeight;
}

double modifier::Value() const
{
	itsResult->ParamIndex(0);
	return itsResult->Value();
}

double modifier::MinimumValue() const
{
	throw kFunctionNotImplemented;
}

double modifier::MaximumValue() const
{
	throw kFunctionNotImplemented;
}

double modifier::Height() const
{
	itsResult->ParamIndex(1);
	return itsResult->Value();
}

double modifier::MaximumHeight() const
{
	throw kFunctionNotImplemented;
}

double modifier::MinimumHeight() const
{
	throw kFunctionNotImplemented;
}

void modifier::Clear(double fillValue)
{
	for (itsResult->ResetTime(); itsResult->NextTime();)
	{
		for (itsResult->ResetLevel(); itsResult->NextLevel();)
		{
			for (itsResult->ResetParam(); itsResult->NextParam();)
			{
				itsResult->Grid()->Data()->Fill(fillValue);
			}
		}
	}
}

bool modifier::IsMissingValue(double theValue) const
{
	if (theValue == kFloatMissing || theValue == kFloatMissing)
	{
		return true;
	}

	return false;
}

void modifier::FindValue(std::shared_ptr<const info> theFindValue)
{
	itsFindValue = std::make_shared<info> (*theFindValue);
	itsFindValue->First();
}

/* ----------------- */

void modifier_max::Calculate(double theValue, double theHeight)
{

	if (IsMissingValue(theValue))
	{
		return;
	}

	itsResult->ParamIndex(0);

	if (IsMissingValue(itsResult->Value()))
	{
		itsResult->Value(theValue);
		itsResult->ParamIndex(1);
		itsResult->Value(theHeight);
	}
	else
	{
		if (theValue > itsResult->Value())
		{
			itsResult->Value(theValue);
			itsResult->ParamIndex(1);
			itsResult->Value(theHeight);
		}
	}
}

double modifier_max::MaximumValue() const
{
	return Value();
}

double modifier_max::MaximumHeight() const
{
	return Height();
}

/* ----------------- */

void modifier_min::Calculate(double theValue, double theHeight)
{
	if (IsMissingValue(theValue))
	{
		return;
	}

	itsResult->ParamIndex(0);

	if (IsMissingValue(itsResult->Value()))
	{
		itsResult->Value(theValue);
		itsResult->ParamIndex(1);
		itsResult->Value(theHeight);
	}
	else
	{
		if (theValue < itsResult->Value())
		{
			itsResult->Value(theValue);
			itsResult->ParamIndex(1);
			itsResult->Value(theHeight);
		}
	}
}

double modifier_min::MinimumValue() const
{
	return Value();
}

double modifier_min::MinimumHeight() const
{
	return Height();
}

/* ----------------- */

void modifier_maxmin::Calculate(double theValue, double theHeight)
{
	if (IsMissingValue(theValue))
	{
		return;
	}

	itsResult->ParamIndex(0); // Max

	// Set min == max
	
	if (IsMissingValue(itsResult->Value()))
	{
		itsResult->Value(theValue);
		itsResult->ParamIndex(1);
		itsResult->Value(theValue);

		itsResult->ParamIndex(2);
		itsResult->Value(theHeight);
		itsResult->ParamIndex(3);
		itsResult->Value(theHeight);
	}
	else
	{
		if (theValue > itsResult->Value())
		{
			itsResult->Value(theValue);
			itsResult->ParamIndex(2);
			itsResult->Value(theHeight);
		}

		itsResult->ParamIndex(1); // Min

		if (theValue < itsResult->Value())
		{
			itsResult->Value(theValue);
			itsResult->ParamIndex(3);
			itsResult->Value(theHeight);
		}

	}
}

double modifier_maxmin::Value() const
{
	throw kFunctionNotImplemented;
}

double modifier_maxmin::Height() const
{
	throw kFunctionNotImplemented;
}

double modifier_maxmin::MinimumValue() const
{
	itsResult->ParamIndex(1);
	return itsResult->Value();
}

double modifier_maxmin::MaximumValue() const
{
	itsResult->ParamIndex(0);
	return itsResult->Value();
}

double modifier_maxmin::MinimumHeight() const
{
	itsResult->ParamIndex(3);
	return itsResult->Value();
}

double modifier_maxmin::MaximumHeight() const
{
	itsResult->ParamIndex(2);
	return itsResult->Value();
}

/* ----------------- */

void modifier_sum::Calculate(double theValue, double theHeight)
{
	if (IsMissingValue(theValue))
	{
		return;
	}

	itsResult->ParamIndex(0); // Max

	if (IsMissingValue(itsResult->Value())) // First value
	{
		itsResult->Value(theValue);
	}
	else
	{
		double val = itsResult->Value();
		itsResult->Value(theValue+val);
	}
}

double modifier_sum::Height() const
{
	throw kFunctionNotImplemented;
}

/* ----------------- */

void modifier_mean::Init(std::shared_ptr<const info> sourceInfo)
{
	itsResult = std::shared_ptr<info> (new info(*sourceInfo));
	itsResult->Create();

	Clear();

	itsResult->First();

	itsValuesCount.resize(itsResult->Grid()->Size());
	
	std::fill(itsValuesCount.begin(), itsValuesCount.end(), kHPMissingInt);

}


void modifier_mean::Calculate(double theValue, double theHeight)
{
	if (IsMissingValue(theValue))
	{
		return;
	}

	modifier_sum::Calculate(theValue, theHeight);

	itsResult->ParamIndex(0);

	if (IsMissingValue(itsResult->Value())) // First value
	{
		//itsResult->Value(theValue);
		itsValuesCount[itsResult->LocationIndex()] = 1;
	}
	else
	{
		//double val = itsResult->Value();
		size_t count = itsValuesCount[itsResult->LocationIndex()];

		//itsResult->Value(theValue+val);
		itsValuesCount[itsResult->LocationIndex()] = count++;
	}
}

std::shared_ptr<info> modifier_mean::Results() const
{
	itsResult->ParamIndex(0);

	for (itsResult->ResetLocation(); itsResult->NextLocation();)
	{
		double val = itsResult->Value();
		size_t count = itsValuesCount[itsResult->LocationIndex()];

		if (val != kFloatMissing && count != static_cast<size_t> (kHPMissingInt))
		{
			itsResult->Value(val / static_cast<double> (count));
		}		
	}

	return itsResult;
}

/* ----------------- */

void modifier_count::Init(std::shared_ptr<const info> sourceInfo)
{
	itsResult = std::shared_ptr<info> (new info(*sourceInfo));
	itsResult->Create();

	Clear(0.);

	itsResult->First();

	itsLowerValueThreshold.resize(itsResult->Grid()->Size());

	std::fill(itsLowerValueThreshold.begin(), itsLowerValueThreshold.end(), kFloatMissing);

}

void modifier_count::Calculate(double theValue, double theHeight)
{
	size_t locationIndex = itsResult->LocationIndex();

	itsFindValue->LocationIndex(locationIndex);

	double findValue = itsFindValue->Value();

	itsResult->ParamIndex(0); // We are interested in the value here

	if (IsMissingValue(theValue) || IsMissingValue(findValue))
	{
		return;
	}

	double lowerValueThreshold = itsLowerValueThreshold[locationIndex];

	// First level

	if (IsMissingValue(lowerValueThreshold))
	{
		itsLowerValueThreshold[locationIndex] = theValue;
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

	// Set lower value already for the next level
	
	itsLowerValueThreshold[locationIndex] = theValue;
	
	if (lowerValueThreshold <= findValue && theValue >= findValue)
	{
		itsResult->Value() == kFloatMissing ? itsResult->Value(1) : itsResult->Value(itsResult->Value() + 1);
	}

	
}

double modifier_count::Height() const
{
	throw kFunctionNotImplemented;
}

/* ----------------- */

bool modifier_findheight::CalculationFinished() const
{
	return itsValuesFound == itsResult->Grid()->Size();
}

void modifier_findheight::Init(std::shared_ptr<const info> sourceInfo)
{
	itsResult = std::shared_ptr<info> (new info(*sourceInfo));
	itsResult->Create();

	Clear();

	itsResult->First();

	itsLowerValueThreshold.resize(itsResult->Grid()->Size());
	itsLowerHeightThreshold.resize(itsResult->Grid()->Size());

	std::fill(itsLowerValueThreshold.begin(), itsLowerValueThreshold.end(), kFloatMissing);
	std::fill(itsLowerHeightThreshold.begin(), itsLowerHeightThreshold.end(), kFloatMissing);

}

void modifier_findheight::Calculate(double theValue, double theHeight)
{

	size_t locationIndex = itsResult->LocationIndex();
	
	itsFindValue->LocationIndex(locationIndex);
	
	double findValue = itsFindValue->Value();
		
	itsResult->ParamIndex(1); // We are interested in the height here

	if (IsMissingValue(theValue) || !IsMissingValue(itsResult->Value()) || IsMissingValue(findValue))
	{
		return;
	}

	if (itsFindNthValue != 1)
	{
		throw std::runtime_error("NthValue other than 1");
	}

	double lowerValueThreshold = itsLowerValueThreshold[locationIndex];
	double lowerHeightThreshold = itsLowerHeightThreshold[locationIndex];

	if (IsMissingValue(lowerValueThreshold))
	{
		itsLowerValueThreshold[locationIndex] = theValue;
		itsLowerHeightThreshold[locationIndex] = theHeight;
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

	itsLowerValueThreshold[locationIndex] = theValue;
	itsLowerHeightThreshold[locationIndex] = theHeight;
	
	if (lowerValueThreshold <= findValue && theValue >= findValue)
	{
		double actualHeight = NFmiInterpolation::Linear(findValue, lowerValueThreshold, theValue, lowerHeightThreshold, theHeight);
		
		itsResult->Value(actualHeight);
		itsResult->ParamIndex(0);
		itsResult->Value(findValue);
	}
	
}

/* ----------------- */

bool modifier_findvalue::CalculationFinished() const
{
	return itsValuesFound == itsResult->Grid()->Size();
}

void modifier_findvalue::Calculate(double theValue, double theHeight)
{

	size_t locationIndex = itsResult->LocationIndex();

	itsFindValue->LocationIndex(locationIndex);

	double findValue = itsFindValue->Value();

	itsResult->ParamIndex(0); // We are interested in the value here

	if (IsMissingValue(theValue) || !IsMissingValue(itsResult->Value()) || IsMissingValue(findValue))
	{
		return;
	}

	if (itsFindNthValue != 1)
	{
		throw std::runtime_error("NthValue other than 1");
	}

	double lowerValueThreshold = itsLowerValueThreshold[locationIndex];
	double lowerHeightThreshold = itsLowerHeightThreshold[locationIndex];

	if (IsMissingValue(lowerValueThreshold))
	{
		itsLowerValueThreshold[locationIndex] = theValue;
		itsLowerHeightThreshold[locationIndex] = theHeight;
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

	itsLowerValueThreshold[locationIndex] = theValue;
	itsLowerHeightThreshold[locationIndex] = theHeight;
	
	if (lowerHeightThreshold <= findValue && theHeight >= findValue)
	{
		double actualValue = NFmiInterpolation::Linear(findValue, lowerHeightThreshold, theHeight, lowerValueThreshold, theValue);

		itsResult->Value(actualValue);
		itsResult->ParamIndex(1);
		itsResult->Value(findValue);
	}
}

void modifier_findvalue::Init(std::shared_ptr<const info> sourceInfo)
{
	itsResult = std::shared_ptr<info> (new info(*sourceInfo));
	itsResult->Create();

	Clear();

	itsResult->First();

	itsLowerValueThreshold.resize(itsResult->Grid()->Size());
	itsLowerHeightThreshold.resize(itsResult->Grid()->Size());

	std::fill(itsLowerValueThreshold.begin(), itsLowerValueThreshold.end(), kFloatMissing);
	std::fill(itsLowerHeightThreshold.begin(), itsLowerHeightThreshold.end(), kFloatMissing);

}