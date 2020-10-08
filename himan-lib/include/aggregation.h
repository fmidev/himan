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

const boost::unordered_map<HPAggregationType, std::string> HPAggregationTypeToString =
    ba::map_list_of(kUnknownAggregationType, "unknown")(kAverage, "average")(kAccumulation, "accumulation")(
        kMaximum, "maximum")(kMinimum, "minimum")(kDifference, "difference");

const boost::unordered_map<std::string, HPAggregationType> HPStringToAggregationType =
    ba::map_list_of("unknown", kUnknownAggregationType)("average", kAverage)("accumulation", kAccumulation)(
        "maximum", kMaximum)("minimum", kMinimum)("difference", kDifference);

class aggregation
{
   public:
	aggregation();
	aggregation(HPAggregationType theAggregationType);
	aggregation(HPAggregationType theAggregationType, const time_duration& theTimeInteval);
	aggregation(HPAggregationType theAggregationType, const time_duration& theTimeInteval,
	            const time_duration& theTimeOffset);

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

#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsType), CEREAL_NVP(itsTimeDuration));
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const aggregation& ob)
{
	return ob.Write(file);
}
}  // namespace himan

#endif /* AGGREGATION_H */
