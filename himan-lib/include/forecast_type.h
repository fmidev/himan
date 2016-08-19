/**
 * @file   forecast_type.h
 * @author partio
 *
 * @date   February 26, 2015, 4:27 PM
 *
 * @brief Simple wrapper for forecast types which are defined in radon database.
 *
 * Current values are:
 *
 * 0: unknown type
 * 1: deterministic forecast
 * 2: analysis
 * 3: eps control forecast
 * 4: eps perturbation
 *
 */

#ifndef FORECAST_TYPE_H
#define FORECAST_TYPE_H

#include "himan_common.h"

namespace himan
{
class forecast_type
{
   public:
	forecast_type() {}
	explicit forecast_type(HPForecastType theType);
	forecast_type(HPForecastType theType, double theValue);

	bool operator==(const forecast_type& theType) const;
	bool operator!=(const forecast_type& theType) const;

	std::string ClassName() const { return "himan::forecast_type"; }
	HPForecastType Type() const;
	void Type(HPForecastType theType);

	double Value() const;
	void Value(double theValue);

	std::ostream& Write(std::ostream& file) const;

   private:
	HPForecastType itsForecastType;
	double itsForecastTypeValue;
};

inline std::ostream& operator<<(std::ostream& file, const forecast_type& ob) { return ob.Write(file); }
}  // namespace himan

#endif /* FORECAST_TYPE_H */
