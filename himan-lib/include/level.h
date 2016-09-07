/**
 * @file level.h
 *
 * @date Nov 30, 2012
 * @author partio
 *
 * @brief Level metadata for himan.
 */

#ifndef LEVEL_H
#define LEVEL_H

#include "himan_common.h"

class NFmiLevel;

namespace himan
{
class logger;
class level
{
   public:
	level();
	explicit level(const NFmiLevel& theLevel);
	level(HPLevelType theType, double theValue);
	level(HPLevelType theType, double theValue, const std::string& theName);
	level(HPLevelType theType, double theValue, double theValue2);

	~level() = default;
	operator std::string() const;

	std::string ClassName() const { return "himan::level"; }
	bool operator==(const level& other) const;
	bool operator!=(const level& other) const;

	void Value(double theLevelValue);
	double Value() const;

	/*
	 * Some levels have two values, like height delta.
	 */

	void Value2(double theLevelValue2);
	double Value2() const;

	void Index(int theIndex);

	int Index() const;

	void Type(HPLevelType theLevelType);

	/**
	 * @return Return level type
	 * @see himan_common.h
	 */

	HPLevelType Type() const;

	std::string Name() const;
	void Name(const std::string& theName);

	std::ostream& Write(std::ostream& file) const;

   private:
	HPLevelType itsType;
	double itsValue;
	double itsValue2;
	int itsIndex;
	std::string itsName;
};

inline std::ostream& operator<<(std::ostream& file, const level& ob) { return ob.Write(file); }
}  // namespace himan

#endif /* LEVEL_H */
