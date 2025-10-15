/**
 * @file aggregation.h
 *
 * @brief simple class to describe *time* aggregation metadata
 */

#ifndef AGGREGATION_H
#define AGGREGATION_H

#include "himan_common.h"
#include "serialization.h"
#include "time_duration.h"

namespace himan
{
enum HPAggregationType
{
	kUnknownAggregationType = 0,
	kAverage,
	kAccumulation,
	kMaximum,
	kMinimum,
	kDifference
};

const std::unordered_map<HPAggregationType, std::string> HPAggregationTypeToString = {
    {kUnknownAggregationType, "unknown"},
    {kAverage, "average"},
    {kAccumulation, "accumulation"},
    {kMaximum, "maximum"},
    {kMinimum, "minimum"},
    {kDifference, "difference"}};

const std::unordered_map<std::string, HPAggregationType> HPStringToAggregationType = {
    {"unknown", kUnknownAggregationType},
    {"average", kAverage},
    {"accumulation", kAccumulation},
    {"maximum", kMaximum},
    {"minimum", kMinimum},
    {"difference", kDifference}};

class aggregation
{
   public:
	aggregation();
	aggregation(HPAggregationType theAggregationType);
	aggregation(HPAggregationType theAggregationType, const time_duration& theTimeInteval);
	aggregation(HPAggregationType theAggregationType, const time_duration& theTimeInteval,
	            const time_duration& theTimeOffset);
	aggregation(const std::string& aggstr);

	~aggregation() = default;
	aggregation(const aggregation&) = default;
	aggregation& operator=(const aggregation&) = default;

	bool operator==(const aggregation& other) const;
	bool operator!=(const aggregation& other) const;
	operator std::string() const;

	std::string ClassName() const
	{
		return "himan::aggregation";
	}
	HPAggregationType Type() const;
	void Type(HPAggregationType theType);

	time_duration TimeDuration() const;
	void TimeDuration(const time_duration& theTimeDuration);

	time_duration TimeOffset() const;
	void TimeOffset(const time_duration& theTimeOffset);

	std::ostream& Write(std::ostream& file) const;

   private:
	HPAggregationType itsType;
	time_duration itsTimeDuration;  // duration of the aggregation period, for exampl *6 hour* precipitation
	time_duration itsTimeOffset;    // offset of the beginning of the aggregation period compared to current hour,
	                                // usually -1 * itsTimeDuration

#ifdef HAVE_CEREAL
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsType), CEREAL_NVP(itsTimeDuration), CEREAL_NVP(itsTimeOffset));
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const aggregation& ob)
{
	return ob.Write(file);
}
}  // namespace himan

template <>
struct fmt::formatter<himan::aggregation>
{
	template <typename ParseContext>
	constexpr auto parse(ParseContext& ctx)
	{
		return ctx.begin();
	}

	template <typename FormatContext>
	auto format(const himan::aggregation& a, FormatContext& ctx) const -> decltype(ctx.out())
	{
		return fmt::format_to(ctx.out(), "{}", static_cast<std::string>(a));
	}
};

#endif /* AGGREGATION_H */
