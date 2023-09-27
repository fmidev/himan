/**
 * @file logger.h
 *
 */

#ifndef LOGGER_H
#define LOGGER_H

#include "himan_common.h"
#include "serialization.h"
#include <fmt/core.h>
#include <fmt/format.h>
#include <stdio.h>

namespace himan
{
class logger
{
   public:
	logger();
	explicit logger(const std::string& theUserName);
	logger(const std::string& theUserName, HPDebugState theDebugState);
	~logger() = default;

	logger(const logger&) = default;
	logger& operator=(const logger&) = default;

	void Trace(const std::string& msg) const
	{
		if (itsDebugState <= kTraceMsg)
		{
			fmt::print("Trace::{} {}\n", itsUserName, msg);
		}
	};

	void Debug(const std::string& msg) const
	{
		if (itsDebugState <= kDebugMsg)
		{
			fmt::print("Debug::{} {}\n", itsUserName, msg);
		}
	};

	void Info(const std::string& msg) const
	{
		if (itsDebugState <= kInfoMsg)
		{
			fmt::print("Info::{} {}\n", itsUserName, msg);
		}
	};

	void Warning(const std::string& msg) const
	{
		if (itsDebugState <= kWarningMsg)
		{
			fmt::print("Warning::{} {}\n", itsUserName, msg);
		}
	};

	void Error(const std::string& msg) const
	{
		if (itsDebugState <= kErrorMsg)
		{
			fmt::print("Error::{} {}\n", itsUserName, msg);
		}
	};

	void Fatal(const std::string& msg) const
	{
		fmt::print("Fatal::{} {}\n", itsUserName, msg);
	};

	static HPDebugState MainDebugState;

   private:
	HPDebugState itsDebugState;
	std::string itsUserName;

#ifdef HAVE_CEREAL
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsDebugState), CEREAL_NVP(itsUserName));
	}
#endif
};

}  // namespace himan

#endif /* LOGGER_H */
