#pragma once
#include "himan_common.h"
#include "serialization.h"
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <fmt/format.h>

#ifdef SERIALIZATION
#include <boost/date_time.hpp>
#include <boost/date_time/posix_time/time_parsers.hpp>

namespace cereal
{
template <class Archive>
inline std::string save_minimal(const Archive& ar, const boost::posix_time::time_duration& td)
{
	return boost::posix_time::to_iso_string(td);
}

template <class Archive>
inline void load_minimal(const Archive& ar, boost::posix_time::time_duration& td, const std::string& str)
{
	if (str == "not-a-date-time")
	{
		td = boost::posix_time::time_duration(boost::posix_time::not_a_date_time);
	}
	else
	{
		td = boost::posix_time::duration_from_string(str);
	}
}
}  // namespace cereal

#endif

namespace himan
{
class time_duration
{
   public:
	time_duration();
	time_duration(HPTimeResolution theTimeResolution, long theTimeResolutionValue);
	time_duration(const boost::posix_time::time_duration& dur);
	time_duration(const std::string& SQLTimeInterval);
	~time_duration() = default;
	time_duration(const time_duration& other) = default;
	operator std::string() const;

	std::string ClassName() const
	{
		return "himan::time_duration";
	}

	/**
	 * @brief Format time duration to string
	 *
	 * Allowed format values:
	 *  %H --> Total hours % 24
	 *  %M --> Total minutes % 60
	 *  %S --> Total seconds % 60
	 *  %d --> Total days
	 *  %h --> Total hours
	 *  %m --> Total minutes
	 *  %s --> Total seconds
	 *
	 * Each value can have further printf-style identifiers, like
	 * %03h --> total hours with up to 3 leading zeros
	 */

	std::string String(const std::string& fmt) const;
	bool operator==(const time_duration&) const;
	bool operator!=(const time_duration&) const;
	bool operator>(const time_duration&) const;
	bool operator<(const time_duration&) const;
	bool operator>=(const time_duration&) const;
	bool operator<=(const time_duration&) const;
	time_duration operator+(const time_duration&) const;
	time_duration& operator+=(const time_duration&);
	time_duration operator-(const time_duration&) const;
	time_duration& operator-=(const time_duration&);
	time_duration operator*(int) const;
	time_duration& operator*=(int);
	time_duration operator/(int) const;
	time_duration& operator/=(int);

	bool Empty() const;

	long Hours() const;
	long Minutes() const;
	long Seconds() const;

	std::ostream& Write(std::ostream& file) const;

   private:
	boost::posix_time::time_duration itsDuration;

#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsDuration));
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const time_duration& ob)
{
	return ob.Write(file);
}

const time_duration FIFTEEN_MINUTES("00:15:00");
const time_duration ONE_HOUR("01:00");
const time_duration THREE_HOURS("03:00");
const time_duration SIX_HOURS("06:00");
const time_duration TWELVE_HOURS("12:00");

}  // namespace himan

template <>
struct fmt::formatter<himan::time_duration>
{
	template <typename ParseContext>
	constexpr auto parse(ParseContext& ctx)
	{
		return ctx.begin();
	}

	template <typename FormatContext>
	auto format(const himan::time_duration& r, FormatContext& ctx) const -> decltype(ctx.out())
	{
		return fmt::format_to(ctx.out(), "{}", static_cast<std::string>(r));
	}
};
