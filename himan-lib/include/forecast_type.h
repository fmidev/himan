/**
 * @file   forecast_type.h
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
#include "serialization.h"

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
	operator std::string() const;

	std::string ClassName() const { return "himan::forecast_type"; }
	HPForecastType Type() const;
	void Type(HPForecastType theType);

	double Value() const;
	void Value(double theValue);

	std::ostream& Write(std::ostream& file) const;

   private:
	HPForecastType itsForecastType;
	double itsForecastTypeValue;

#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsForecastType), CEREAL_NVP(itsForecastTypeValue));
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const forecast_type& ob) { return ob.Write(file); }
}  // namespace himan

#endif /* FORECAST_TYPE_H */
