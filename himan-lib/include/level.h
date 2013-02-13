/**
 * @file level.h
 *
 * @date Nov 30, 2012
 * @author partio
 *
 * @brief Level metadata for himan. Uses NFmiLevel for some features.
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
    level(HPLevelType theType, float theValue, const std::string& theName = "");
    level(HPLevelType theType, float theValue, int theIndex, const std::string& theName = "");

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

    /**
     * @brief Set level values (for pressure levels)
     */

    void Value(float theLevelValue);

    /**
     * @return Level value (for pressure levels)
     */
    float Value() const;

    /**
     * @brief Set level index number
     */
    void Index(int theIndex);

    /**
     * @return Level index number
     */

    int Index() const;

    //void Type(HPLevelType theType);

    /**
     * @return Level type
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

    std::unique_ptr<NFmiLevel> itsLevel;
    std::unique_ptr<logger> itsLogger;
    int itsIndex;
};

inline
std::ostream& operator<<(std::ostream& file, const level& ob)
{
    return ob.Write(file);
}

} // namespace himan

#endif /* LEVEL_H */
