/*
 * level.cpp
 *
 */

#include "level.h"
#include "NFmiLevel.h"

#include <ostream>

using namespace himan;

level::level()
    : itsType(kUnknownLevel), itsValue(kHPMissingValue), itsValue2(kHPMissingValue), itsIndex(kHPMissingInt), itsName()
{
}

level::level(HPLevelType theType, double theValue)
    : itsType(theType), itsValue(theValue), itsValue2(kHPMissingValue), itsIndex(kHPMissingInt), itsName()
{
}

level::level(HPLevelType theType, double theValue, const std::string& theName)
    : itsType(theType), itsValue(theValue), itsValue2(kHPMissingValue), itsIndex(kHPMissingInt), itsName()
{
}

level::level(HPLevelType theType, double theValue, double theValue2)
    : itsType(theType), itsValue(theValue), itsValue2(theValue2), itsIndex(kHPMissingInt), itsName()
{
}

bool level::operator==(const level& other) const
{
	if (this == &other)
	{
		return true;
	}

	return (itsType == other.itsType && itsValue == other.itsValue && itsValue2 == other.itsValue2);
}

bool level::operator!=(const level& other) const { return !(*this == other); }
level::operator std::string() const
{
	std::string out = HPLevelTypeToString.at(itsType) + "/" + std::to_string(itsValue);

	if (!IsKHPMissingValue(itsValue2))
	{
		out += "/" + std::to_string(itsValue2);
	}

	return out;
}

void level::Value(double theValue) { itsValue = theValue; }
double level::Value() const { return itsValue; }
void level::Value2(double theValue2) { itsValue2 = theValue2; }
double level::Value2() const { return itsValue2; }
int level::Index() const { return itsIndex; }
void level::Index(int theIndex) { itsIndex = theIndex; }
HPLevelType level::Type() const { return itsType; }
void level::Type(HPLevelType theLevelType) { itsType = theLevelType; }
std::string level::Name() const { return itsName; }
void level::Name(const std::string& theName) { itsName = theName; }
std::ostream& level::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;
	file << "__itsType__ " << HPLevelTypeToString.at(itsType) << std::endl;
	file << "__itsIndex__ " << itsIndex << std::endl;
	file << "__itsValue__ " << itsValue << std::endl;
	file << "__itsValue2__ " << itsValue2 << std::endl;
	file << "__itsName__ " << itsName << std::endl;

	return file;
}
