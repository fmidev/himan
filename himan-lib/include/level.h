/**
 * @file level.h
 *
 * @brief Level metadata for himan.
 */

#ifndef LEVEL_H
#define LEVEL_H

#include "himan_common.h"
#include "serialization.h"

namespace himan
{

class level
{
   public:
	level();
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

	void Value2(double theLevelValue2);
	double Value2() const;

	void Index(int theIndex);

	int Index() const;

	void Type(HPLevelType theLevelType);
	HPLevelType Type() const;

	std::string Name() const;
	void Name(const std::string& theName);

	std::ostream& Write(std::ostream& file) const;

   private:
	HPLevelType itsType;

	/*
	 * itsValue variable contains the value of the level (doh).
	 * In the majority of the cases, a level has only single value
	 * and it's stored here.
	 */
	double itsValue;

	/*
	 * itsValue2 contains the _possible_ second value related to the
	 * level. This is used for example for a level that's actually a
	 * layer between two height values. In this case a common interpretation
	 * is that 'itsValue' is the upper level value of the layer, and
	 * 'itsValue2' is the lower level value of the layer.
	 *
	 * The variable is ambiguosly named on purpose, because:
	 * - In almost all cases, we only have one level value which is not
	 *   either high or low. In this case itsValue2 is missing value.
	 * - In some cases it could be that the two values are not high and
	 *   low but something else. Currently all layers between two levels
	 *   are defined with top/bottom, but that might not be the case in
	 *   the future.
	 */

	double itsValue2;
	int itsIndex;  // Level index, ie. the number of level in a file for example
	std::string itsName;

#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsType), CEREAL_NVP(itsValue), CEREAL_NVP(itsValue2), CEREAL_NVP(itsIndex), CEREAL_NVP(itsName));
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const level& ob) { return ob.Write(file); }
}  // namespace himan

#endif /* LEVEL_H */
