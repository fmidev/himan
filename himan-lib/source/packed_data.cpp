/**
 * @file packed_data.cpp
 *
 * @date Apr 11, 2013
 * @author partio
 */

#include "packed_data.h"

simple_packed::simple_packed(int theBitsPerValue, double theBinaryScaleFactor, double theDecimalScaleFactor, double theReferenceValue)
	: itsBitsPerValue(theBitsPerValue)
	, itsBinaryScaleFactor(theBinaryScaleFactor)
	, itsDecimalScaleFactor(theDecimalScaleFactor)
	, itsReferenceValue(theReferenceValue)
{}

int simple_packed::BitsPerValue() const
{
	return itsBitsPerValue;
}

double simple_packed::BinaryScaleFactor() const
{
	return itsBinaryScaleFactor;
}

double simple_packed::DecimalScaleFactor() const
{
	return itsDecimalScaleFactor;
}

double simple_packed::ReferenceValue() const
{
	return itsReferenceValue;
}