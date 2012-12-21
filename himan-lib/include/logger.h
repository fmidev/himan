/*
 * logger.h
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 */

#ifndef LOGGER_H
#define LOGGER_H

#include "hilpee_common.h"
#include <string>
#include <stdio.h>
#include <memory> // for std::unique_ptr

namespace hilpee
{

class logger
{
	public:
		~logger() {}

		logger(const std::string& theUserName, HPDebugState theDebugState);

		void Trace(const std::string& msg)
		{
			if (itsDebugState <= kTraceMsg)
			{
				printf("Trace <%s> %s\n", itsUserName.c_str(), msg.c_str());
			}
		};

		void Debug(const std::string& msg)
		{
			if (itsDebugState <= kDebugMsg)
			{
				printf("Debug <%s> %s\n", itsUserName.c_str(), msg.c_str());
			}

		};

		void Info(const std::string& msg)
		{
			if (itsDebugState <= kInfoMsg)
			{
				printf("Info <%s> %s\n", itsUserName.c_str(), msg.c_str());
			}

		};

		void Warning(const std::string& msg)
		{

			if (itsDebugState <= kWarningMsg)
			{
				printf("Warning <%s> %s\n", itsUserName.c_str(), msg.c_str());
			}

		};

		void Error(const std::string& msg)
		{
			if (itsDebugState <= kErrorMsg)
			{
				printf("Error <%s> %s\n", itsUserName.c_str(), msg.c_str());
			}

		};

		void Fatal(const std::string& msg)
		{

			if (itsDebugState <= kFatalMsg)
			{
				printf("Fatal <%s> %s\n", itsUserName.c_str(), msg.c_str());
			}

		};

	private:
		HPDebugState itsDebugState;
		std::string itsUserName;

};

} // namespace hilpee

#endif /* LOGGER_H */
