/**
 * @file raw_time.h
 *
 *
 * @brief Wrapper for boost::posix_time.
 *
 */

#ifndef RAW_TIME_H
#define RAW_TIME_H

#include "himan_common.h"
#include "serialization.h"
#include "time_duration.h"
#include <boost/date_time.hpp>
#include <fmt/format.h>

#ifdef HAVE_CEREAL
namespace cereal
{
template <class Archive>
inline std::string save_minimal(const Archive& ar, const boost::posix_time::ptime& pt)
{
	// from_iso_string discards fractional seconds, but that doesn't concern
	// us because we don't have them to begin with.
	// http://www.boost.org/doc/libs/master/doc/html/date_time/posix_time.html

	return boost::posix_time::to_iso_string(pt);
}

template <class Archive>
inline void load_minimal(const Archive& ar, boost::posix_time::ptime& pt, const std::string& str)
{
	pt = boost::posix_time::from_iso_string(str);
}
}  // namespace cereal
#endif

namespace himan
{
class logger;

class raw_time
{
   public:
	friend class forecast_time;

	raw_time() : itsDateTime(boost::posix_time::not_a_date_time)
	{
	}
	raw_time(const boost::posix_time::ptime& ptime);
	raw_time(const std::string& theTime, const std::string& theTimeMask = "%Y-%m-%d %H:%M:%S");
	~raw_time() = default;

	raw_time(const raw_time& other);
	raw_time& operator=(const raw_time& other);
	operator std::string() const;
	raw_time operator+(const time_duration& duration) const;
	raw_time operator-(const time_duration& duration) const;
	raw_time& operator+=(const time_duration& duration);
	raw_time& operator-=(const time_duration& duration);
	time_duration operator-(const raw_time& other) const;

	std::string String(const std::string& theTimeMask = "%Y-%m-%d %H:%M:%S") const;

	std::ostream& Write(std::ostream& file) const;

	std::string ClassName() const
	{
		return "himan::raw_time";
	}
	bool operator==(const raw_time& other) const;
	bool operator!=(const raw_time& other) const;
	bool operator>(const raw_time& other) const;
	bool operator<(const raw_time& other) const;
	bool operator>=(const raw_time& other) const;
	bool operator<=(const raw_time& other) const;

	void Adjust(HPTimeResolution timeResolution, int theValue);

	bool Empty() const;
	bool IsLeapYear() const;

	std::string ToDatabaseTime() const;
	std::string ToDate() const;
	std::string ToTime() const;
	std::string ToSQLTime() const;

	static raw_time Now();
	static raw_time UtcNow();

   private:
	std::string FormatTime(const std::string& theTimeMask) const;

	void FromDatabaseTime(const std::string& databaseTime);

	void FromSQLTime(const std::string& SQLTime);

	boost::posix_time::ptime itsDateTime;

#ifdef HAVE_CEREAL
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsDateTime));
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const raw_time& ob)
{
	return ob.Write(file);
}
}  // namespace himan

template <>
struct fmt::formatter<himan::raw_time>
{
	template <typename ParseContext>
	constexpr auto parse(ParseContext& ctx)
	{
		return ctx.begin();
	}

	template <typename FormatContext>
	auto format(const himan::raw_time& r, FormatContext& ctx) const -> decltype(ctx.out())
	{
		return fmt::format_to(ctx.out(), "{}", static_cast<std::string>(r));
	}
};

#endif /* RAW_TIME_H */
