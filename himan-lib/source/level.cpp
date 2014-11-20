/*
 * level.cpp
 *
 *  Created on: Nov 27, 2012
 *      Author: partio
 */

#include "level.h"
#include <ostream>
#include "NFmiLevel.h"
#include <boost/lexical_cast.hpp>

using namespace himan;

level::level()
    : itsType(kUnknownLevel), itsValue(kHPMissingValue), itsIndex(kHPMissingInt), itsName("")
{
}

level::level(HPLevelType theType, double theValue, const std::string& theName)
    : itsType(theType)
	, itsValue(theValue)
    , itsIndex(kHPMissingInt)
	, itsName(theName)
{
}

level::level(HPLevelType theType, double theValue, int theIndex, const std::string& theName)
    : itsType(theType)
	, itsValue(theValue)
    , itsIndex(theIndex)
	, itsName(theName)
{
}

level::level(const NFmiLevel& theLevel)
    : itsType(static_cast<HPLevelType> (theLevel.LevelType()))
	, itsValue(static_cast<double> (theLevel.LevelValue()))
    , itsIndex(kHPMissingInt)
{
}

level::level(const level& other)
    : itsType(other.itsType)
	, itsValue(other.itsValue)
    , itsIndex(other.itsIndex)
	, itsName(other.itsName)
{
}

level& level::operator=(const level& other)
{
    itsType = other.itsType;
	itsValue = other.itsValue;
	itsIndex = other.itsIndex;
	itsName = other.itsName;
	
    return *this;
}


bool level::operator==(const level& other) const
{
    if (this == &other)
    {
        return true;
    }

    return (itsType == other.itsType && itsValue == other.itsValue);
}

bool level::operator!=(const level& other) const
{
    return !(*this == other);
}

level::operator std::string () const
{
	return static_cast<std::string> (HPLevelTypeToString.at(itsType)) + "/" + boost::lexical_cast<std::string> (itsValue);
}

void level::Value(double theValue)
{
    itsValue = theValue;
}

double level::Value() const
{
    return itsValue;
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

std::ostream& level::Write(std::ostream& file) const
{

    file << "<" << ClassName() << ">" << std::endl;
    file << "__itsType__ " << HPLevelTypeToString.at(itsType) << std::endl;
    file << "__itsIndex__ " << itsIndex << std::endl;
    file << "__itsValue__ " << itsValue << std::endl;
    file << "__itsName__ " << itsName << std::endl;

    return file;
}
