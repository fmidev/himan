/*
 * forecast_time.h
 *
 *  Created on: Dec 1, 2012
 *      Author: partio
 *
 * Container for raw_times.
 */

#ifndef FORECAST_TIME_H
#define FORECAST_TIME_H

#include "logger.h"
#include "raw_time.h"
#include <stdexcept>

namespace hilpee
{

class forecast_time
{

	public:
		forecast_time();
		forecast_time(const raw_time& theOriginDateTime, const raw_time& theValidDateTime);
		forecast_time(std::shared_ptr<raw_time> theOriginDateTime, std::shared_ptr<raw_time> theValidDateTime);
		forecast_time(const std::string& theOriginDateTime,
		              const std::string& theValidDateTime,
		              const std::string& theDateMask = "%Y-%m-%d %H:%M:%S");

		std::string ClassName() const
		{
			return "hilpee::forecast_time";
		};

		HPVersionNumber Version() const
		{
			return HPVersionNumber(0, 1);
		}

		~forecast_time() {}

		std::ostream& Write(std::ostream& file) const;

        bool operator==(const forecast_time& other);
        bool operator!=(const forecast_time& other);

		int Step() const;

		std::shared_ptr<raw_time> OriginDateTime() const;
		void OriginDateTime(std::shared_ptr<raw_time> theOriginDateTime);
		void OriginDateTime(std::string& theOriginDateTime, const std::string& theDateMask = "%Y-%m-%d %H:%M:%S");

		std::shared_ptr<raw_time> ValidDateTime() const;
		void ValidDateTime(std::shared_ptr<raw_time> theValidDateTime);
		void ValidDateTime(std::string& theValidDateTime, const std::string& theDateMask = "%Y-%m-%d %H:%M:%S");

	private:
		std::shared_ptr<logger> itsLogger; // using shared instead of unique since unique prevents
		// copying of object

		std::shared_ptr<raw_time> itsOriginDateTime;
		std::shared_ptr<raw_time> itsValidDateTime;

};

inline
std::ostream& operator<<(std::ostream& file, forecast_time& ob)
{
	return ob.Write(file);
}

} // namespace hilpee

#endif /* FORECAST_TIME_H */
