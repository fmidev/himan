/*
 * level.cpp
 *
 *  Created on: Nov 27, 2012
 *      Author: partio
 */

#include "level.h"
#include "logger_factory.h"
#include <ostream>

using namespace himan;

level::level()
	: itsIndex(kHPMissingInt)
{
	itsLogger = logger_factory::Instance()->GetLog("level");
}

level::level(HPLevelType theType, float theValue)
	: itsLevel(std::unique_ptr<NFmiLevel> (new NFmiLevel(static_cast<FmiLevelType> (theType), theValue)))
	, itsIndex(kHPMissingInt)
{
	itsLogger = logger_factory::Instance()->GetLog("level");
}

level::level(HPLevelType theType, float theValue, int theIndex)
	: itsLevel(std::unique_ptr<NFmiLevel> (new NFmiLevel(static_cast<FmiLevelType> (theType), theValue)))
	, itsIndex(theIndex)
{
	itsLogger = logger_factory::Instance()->GetLog("level");
}

level::level(const NFmiLevel& theLevel)
	: itsLevel(std::unique_ptr<NFmiLevel> (new NFmiLevel(theLevel)))
	, itsIndex(kHPMissingInt)
{
	itsLogger = logger_factory::Instance()->GetLog("level");
}

level::level(const level& other)
	: itsLevel(std::unique_ptr<NFmiLevel> (new NFmiLevel(*other.itsLevel)))
	, itsIndex(other.itsIndex)
{
	itsLogger = logger_factory::Instance()->GetLog("level");
}

level& level::operator=(const level& other)
{
	itsLevel = std::unique_ptr<NFmiLevel> (new NFmiLevel(*other.itsLevel));
	itsIndex = other.itsIndex;

	return *this;
}


bool level::operator==(const level& other)
{
	if (this == &other)
	{
		return true;
	}

	return (Type() == other.Type() && Value() == other.Value());
}

bool level::operator!=(const level& other)
{
	return !(*this == other);
}

void level::Value(float theValue)
{
	itsLevel->LevelValue(theValue);
}

float level::Value() const
{
	return itsLevel->LevelValue();
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
	return HPLevelType(itsLevel->LevelType());
}
/*
void level::Type(HPLevelType theLevelType)
{
	itsLevel->LevelType(theLevelType);
}
*/

std::ostream& level::Write(std::ostream& file) const
{

	file << "<" << ClassName() << " " << Version() << ">" << std::endl;
	file << "__itsType__ " << itsLevel->LevelType() << std::endl;
	file << "__itsIndex__ " << itsIndex << std::endl;
	file << "__itsValue__ " << itsLevel->LevelValue() << std::endl;

	return file;
}
