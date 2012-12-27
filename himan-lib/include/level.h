/*
 * level.h
 *
 *  Created on: Nov 30, 2012
 *      Author: partio
 *
 * Overcoat for NFmiLevel
 */

#ifndef LEVEL_H
#define LEVEL_H

#include "logger.h"
#include "NFmiLevel.h"

namespace himan
{

class level
{

	public:

		level();
		level(const NFmiLevel& theLevel);
		level(HPLevelType theType, float theValue);
		level(HPLevelType theType, float theValue, int theIndex);

		~level() {}
		level(const level& other);
		level& operator=(const level& other);

		std::string ClassName() const
		{
			return "himan::level";
		}

		HPVersionNumber Version() const
		{
			return HPVersionNumber(0, 1);
		}

		bool operator==(const level& other);
		bool operator!=(const level& other);

		void Value(float theLevelValue);
		float Value() const;

		void Index(int theIndex);
		int Index() const;

		//void Type(HPLevelType theType);
		HPLevelType Type() const;

		std::ostream& Write(std::ostream& file) const;

	private:

		std::unique_ptr<NFmiLevel> itsLevel;
		std::unique_ptr<logger> itsLogger;
		int itsIndex;
};

inline
std::ostream& operator<<(std::ostream& file, level& ob)
{
	return ob.Write(file);
}

} // namespace himan

#endif /* LEVEL_H */
