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
    level(const NFmiLevel& theLevel);
    level(HPLevelType theType, double theValue, const std::string& theName = "");
    level(HPLevelType theType, double theValue, int theIndex, const std::string& theName = "");

    ~level() = default;
    level(const level& other);
    level& operator=(const level& other);
	operator std::string () const;

    std::string ClassName() const
    {
        return "himan::level";
    }

    bool operator==(const level& other) const;
    bool operator!=(const level& other) const;

    /**
     * @brief Set level values (for pressure levels)
     */

    void Value(double theLevelValue);

    /**
     * @return Level value (for pressure levels)
     */

    double Value() const;

    /**
     * @brief Set level index number
     */
    void Index(int theIndex);

    /**
     * @return Level index number
     */

    int Index() const;

	/**
	 * @brief Set Level type
     * @param theType
     */
	
    void Type(HPLevelType theLevelType);

    /**
     * @return Return level type
     * @see himan_common.h
     */

    HPLevelType Type() const;

    /**
     * @brief deprecated
     */

    std::string Name() const;

    /**
     * @brief deprecated
     */

    void Name(const std::string& theName);

    std::ostream& Write(std::ostream& file) const;

private:
	std::unique_ptr<logger> itsLogger;

    HPLevelType itsType;
	double itsValue;
    int itsIndex;
	std::string itsName;
};

inline
std::ostream& operator<<(std::ostream& file, const level& ob)
{
    return ob.Write(file);
}

} // namespace himan

#endif /* LEVEL_H */
