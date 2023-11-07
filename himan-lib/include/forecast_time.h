/**
 * @file forecast_time.h
 *
 */

#ifndef FORECAST_TIME_H
#define FORECAST_TIME_H

#include "raw_time.h"
#include "serialization.h"

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
	forecast_time() = default;
	forecast_time(const raw_time& theOriginDateTime, const raw_time& theValidDateTime);
	forecast_time(const std::string& theOriginDateTime, const std::string& theValidDateTime,
	              const std::string& theDateMask = "%Y-%m-%d %H:%M:%S");
	forecast_time(const raw_time& theOriginDateTime, const time_duration& theStep);

	~forecast_time() = default;
	forecast_time(const forecast_time&) = default;
	forecast_time& operator=(const forecast_time&) = default;

	std::string ClassName() const
	{
		return "himan::forecast_time";
	}
	std::ostream& Write(std::ostream& file) const;

	bool operator==(const forecast_time& other) const;
	bool operator!=(const forecast_time& other) const;

	const raw_time& OriginDateTime() const;
	raw_time& OriginDateTime();
	void OriginDateTime(const raw_time& theOriginDateTime);
	void OriginDateTime(const std::string& theOriginDateTime, const std::string& theDateMask = "%Y-%m-%d %H:%M:%S");

	const raw_time& ValidDateTime() const;
	raw_time& ValidDateTime();
	void ValidDateTime(const raw_time& theValidDateTime);
	void ValidDateTime(const std::string& theValidDateTime, const std::string& theDateMask = "%Y-%m-%d %H:%M:%S");

	time_duration Step() const;

   private:
	raw_time itsOriginDateTime;
	raw_time itsValidDateTime;

#ifdef HAVE_CEREAL
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsOriginDateTime), CEREAL_NVP(itsValidDateTime));
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const forecast_time& ob)
{
	return ob.Write(file);
}
}  // namespace himan

template <>
struct fmt::formatter<himan::forecast_time>
{
	template <typename ParseContext>
	constexpr auto parse(ParseContext& ctx)
	{
		return ctx.begin();
	}

	template <typename FormatContext>
	auto format(const himan::forecast_time& ft, FormatContext& ctx) const -> decltype(ctx.out())
	{
		return fmt::format_to(ctx.out(), "{} step: {}", ft.OriginDateTime(), ft.Step());
	}
};

#endif /* FORECAST_TIME_H */
