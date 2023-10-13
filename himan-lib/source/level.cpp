/*
 * level.cpp
 *
 */

#include "level.h"
#include <fmt/core.h>
#include <ostream>

using namespace himan;

level::level()
    : itsType(kUnknownLevel),
      itsValue(kHPMissingValue),
      itsValue2(kHPMissingValue),
      itsIndex(kHPMissingInt),
      itsName(),
      itsAB()
{
}

level::level(HPLevelType theType, double theValue)
    : itsType(theType), itsValue(theValue), itsValue2(kHPMissingValue), itsIndex(kHPMissingInt), itsName(), itsAB()
{
}

level::level(HPLevelType theType, double theValue, const std::string& theName)
    : itsType(theType),
      itsValue(theValue),
      itsValue2(kHPMissingValue),
      itsIndex(kHPMissingInt),
      itsName(theName),
      itsAB()
{
}

level::level(HPLevelType theType, double theValue, double theValue2)
    : itsType(theType), itsValue(theValue), itsValue2(theValue2), itsIndex(kHPMissingInt), itsName(), itsAB()
{
}

bool level::operator==(const level& other) const
{
	if (this == &other)
	{
		return true;
	}

	return (itsType == other.itsType && itsValue == other.itsValue && itsValue2 == other.itsValue2);  // &&
	//	        itsAB == other.itsAB);
}

bool level::operator!=(const level& other) const
{
	return !(*this == other);
}
level::operator std::string() const
{
	std::string out = fmt::format("{}/{}", HPLevelTypeToString.at(itsType), itsValue);

	if (!IsKHPMissingValue(itsValue2))
	{
		out = fmt::format("{}/{}", out, itsValue2);
	}

	return out;
}

void level::EqualAdjustment(level& lev, double adj)
{
	switch (lev.Type())
	{
		default:
			lev.Value() += adj;
			break;
		case kGeneralizedVerticalLayer:
		case kGroundDepth:
		case kHeightLayer:
			lev.Value() += adj;
			lev.Value2() += adj;
			break;
	}
}
void level::Value(double theValue)
{
	itsValue = theValue;
}
double level::Value() const
{
	return itsValue;
}
double& level::Value()
{
	return itsValue;
}
void level::Value2(double theValue2)
{
	itsValue2 = theValue2;
}
double level::Value2() const
{
	return itsValue2;
}
double& level::Value2()
{
	return itsValue2;
}

int level::Index() const
{
	return itsIndex;
}
void level::Index(int theIndex)
{
	itsIndex = theIndex;
}
HPLevelType level::Type() const
{
	return itsType;
}
void level::Type(HPLevelType theLevelType)
{
	itsType = theLevelType;
}
std::string level::Name() const
{
	return itsName;
}
void level::Name(const std::string& theName)
{
	itsName = theName;
}
std::vector<double> level::AB() const
{
	return itsAB;
}
void level::AB(const std::vector<double>& theAB)
{
	itsAB = theAB;
}
std::ostream& level::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;
	file << "__itsType__ " << HPLevelTypeToString.at(itsType) << std::endl;
	file << "__itsIndex__ " << itsIndex << std::endl;
	file << "__itsValue__ " << itsValue << std::endl;
	file << "__itsValue2__ " << itsValue2 << std::endl;
	file << "__itsName__ " << itsName << std::endl;
	file << "__itsAB__ " << itsAB.size() << std::endl;

	return file;
}
