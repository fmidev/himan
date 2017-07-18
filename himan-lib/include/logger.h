/**
 * @file logger.h
 *
 */

#ifndef LOGGER_H
#define LOGGER_H

#include "himan_common.h"
#include "serialization.h"
#include <stdio.h>

namespace himan
{
class logger
{
   public:
	logger();
	explicit logger(const std::string& theUserName);
	logger(const std::string& theUserName, HPDebugState theDebugState);
	~logger() {}

	logger(const logger& other) = default;
	logger& operator=(const logger& other) = default;

	void Trace(const std::string& msg) const
	{
		if (itsDebugState <= kTraceMsg)
		{
			printf("Trace::%s %s\n", itsUserName.c_str(), msg.c_str());
		}
	};

	void Debug(const std::string& msg) const
	{
		if (itsDebugState <= kDebugMsg)
		{
			printf("Debug::%s %s\n", itsUserName.c_str(), msg.c_str());
		}
	};

	void Info(const std::string& msg) const
	{
		if (itsDebugState <= kInfoMsg)
		{
			printf("Info::%s %s\n", itsUserName.c_str(), msg.c_str());
		}
	};

	void Warning(const std::string& msg) const
	{
		if (itsDebugState <= kWarningMsg)
		{
			printf("Warning::%s %s\n", itsUserName.c_str(), msg.c_str());
		}
	};

	void Error(const std::string& msg) const
	{
		if (itsDebugState <= kErrorMsg)
		{
			printf("Error::%s %s\n", itsUserName.c_str(), msg.c_str());
		}
	};

	void Fatal(const std::string& msg) const { printf("Fatal::%s %s\n", itsUserName.c_str(), msg.c_str()); };

	static HPDebugState MainDebugState;

   private:
	HPDebugState itsDebugState;
	std::string itsUserName;

#ifdef SERIALIZATION
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
