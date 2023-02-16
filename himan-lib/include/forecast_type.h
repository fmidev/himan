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
#include <fmt/format.h>

namespace himan
{
enum HPForecastType
{
	kUnknownType = 0,
	kDeterministic,
	kAnalysis,
	kEpsPerturbation = 3,
	kEpsControl = 4,
	kStatisticalProcessing = 5
};

const boost::unordered_map<HPForecastType, std::string> HPForecastTypeToString = ba::map_list_of(
    kUnknownType, "unknown")(kDeterministic, "deterministic")(kAnalysis, "analysis")(kEpsControl, "eps control")(
    kEpsPerturbation, "eps perturbation")(kStatisticalProcessing, "statistical post processing");

const boost::unordered_map<std::string, HPForecastType> HPStringToForecastType =
    ba::map_list_of("unknown", kUnknownType)("deterministic", kDeterministic)("analysis", kAnalysis)(
        "eps control", kEpsControl)("eps perturbation", kEpsPerturbation);

class forecast_type
{
   public:
	forecast_type() = default;
	explicit forecast_type(HPForecastType theType);
	forecast_type(HPForecastType theType, double theValue);

	bool operator==(const forecast_type& theType) const;
	bool operator!=(const forecast_type& theType) const;
	operator std::string() const;

	std::string ClassName() const
	{
		return "himan::forecast_type";
	}
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

inline std::ostream& operator<<(std::ostream& file, const forecast_type& ob)
{
	return ob.Write(file);
}
}  // namespace himan

template <>
struct fmt::formatter<himan::forecast_type>
{
	template <typename ParseContext>
	constexpr auto parse(ParseContext& ctx)
	{
		return ctx.begin();
	}

	template <typename FormatContext>
	auto format(const himan::forecast_type& ft, FormatContext& ctx) const -> decltype(ctx.out())
	{
		return fmt::format_to(ctx.out(), "{}", static_cast<std::string>(ft));
	}
};

#endif /* FORECAST_TYPE_H */
