#pragma once

#include "radon_record.h"
#include "file_information.h"
#include "param.h"
#include "forecast_type.h"
#include "producer.h"
#include "forecast_time.h"
#include "level.h"

namespace himan
{
struct summary_record
{
	file_information finfo;
	plugin::radon_record rrecord;
	himan::producer producer;
	himan::forecast_type ftype;
	himan::forecast_time ftime;
	himan::level level;
	himan::param param;

	summary_record() = default;
	summary_record(const file_information& finfo_, const plugin::radon_record& rrecord_, const himan::producer& producer_,
	               const himan::forecast_type ftype_, const himan::forecast_time ftime_, const himan::level level_,
	               const himan::param param_)
	    : finfo(finfo_),
	      rrecord(rrecord_),
	      producer(producer_),
	      ftype(ftype_),
	      ftime(ftime_),
	      level(level_),
	      param(param_)
	{
	}
};
} // namespace himan
