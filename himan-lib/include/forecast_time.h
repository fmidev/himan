/**
 * @file forecast_time.h
 *
 * @date Dec 1, 2012
 * @author partio
 *
 */

#ifndef FORECAST_TIME_H
#define FORECAST_TIME_H

#include "raw_time.h"

namespace himan
{

/**
 * @class forecast_time
 *
 * @brief Describe a forecast time: origin time and valid (forecasted) time
 */

class logger;
class forecast_time
{

public:
    forecast_time();
    forecast_time(const raw_time& theOriginDateTime, const raw_time& theValidDateTime);
    forecast_time(const std::string& theOriginDateTime,
                  const std::string& theValidDateTime,
                  const std::string& theDateMask = "%Y-%m-%d %H:%M:%S");

    ~forecast_time() = default;
    forecast_time(const forecast_time& other);
    forecast_time& operator=(const forecast_time& other);

    std::string ClassName() const
    {
        return "himan::forecast_time";
    };

    std::ostream& Write(std::ostream& file) const;

    bool operator==(const forecast_time& other) const;
    bool operator!=(const forecast_time& other) const;

    int Step() const;

    raw_time& OriginDateTime();
    void OriginDateTime(const raw_time& theOriginDateTime);
    void OriginDateTime(const std::string& theOriginDateTime, const std::string& theDateMask = "%Y-%m-%d %H:%M:%S");

    raw_time& ValidDateTime();
    void ValidDateTime(const raw_time& theValidDateTime);
    void ValidDateTime(const std::string& theValidDateTime, const std::string& theDateMask = "%Y-%m-%d %H:%M:%S");

    /**
	 *
     * @return Time step resolution
     */

    HPTimeResolution StepResolution() const;
    void StepResolution(HPTimeResolution theStepResolution);

private:
    raw_time itsOriginDateTime;
    raw_time itsValidDateTime;

    HPTimeResolution itsStepResolution;
};

inline
std::ostream& operator<<(std::ostream& file, const forecast_time& ob)
{
    return ob.Write(file);
}

} // namespace himan

#endif /* FORECAST_TIME_H */
