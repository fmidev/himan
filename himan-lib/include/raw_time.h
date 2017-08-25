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
#include <boost/date_time.hpp>

#ifdef SERIALIZATION
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

	raw_time() : itsDateTime(boost::posix_time::not_a_date_time) {}
	raw_time(const std::string& theTime, const std::string& theTimeMask = "%Y-%m-%d %H:%M:%S");

	~raw_time() {}
	raw_time(const raw_time& other);
	raw_time& operator=(const raw_time& other);
	operator std::string() const;

	std::string String(const std::string& theTimeMask = "%Y-%m-%d %H:%M:%S") const;

	std::ostream& Write(std::ostream& file) const;

	std::string ClassName() const { return "himan::raw_time"; }
	bool operator==(const raw_time& other) const;
	bool operator!=(const raw_time& other) const;

	bool Adjust(HPTimeResolution timeResolution, int theValue);

	bool Empty() const;
	bool IsLeapYear() const;

   private:
	std::string FormatTime(const std::string& theTimeMask) const;
	std::string ToNeonsTime() const;
	void FromNeonsTime(const std::string& neonsTime);

	std::string ToSQLTime() const;
	void FromSQLTime(const std::string& SQLTime);

	boost::posix_time::ptime itsDateTime;

#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsDateTime));
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const raw_time& ob) { return ob.Write(file); }
}  // namespace himan

#endif /* RAW_TIME_H */
