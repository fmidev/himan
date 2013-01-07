/*
 * logger.h
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 */

#ifndef LOGGER_H
#define LOGGER_H

#include "himan_common.h"
#include <string>
#include <stdio.h>

namespace himan
{

class logger
{
public:
    ~logger() {}

    logger(const std::string& theUserName, HPDebugState theDebugState);

    inline void Trace(const std::string& msg)
    {
        if (itsDebugState <= kTraceMsg)
        {
            printf("Trace::%s %s\n", itsUserName.c_str(), msg.c_str());
        }
    };

    inline void Debug(const std::string& msg)
    {
        if (itsDebugState <= kDebugMsg)
        {
            printf("Debug::%s %s\n", itsUserName.c_str(), msg.c_str());
        }

    };

    inline void Info(const std::string& msg)
    {
        if (itsDebugState <= kInfoMsg)
        {
            printf("Info::%s %s\n", itsUserName.c_str(), msg.c_str());
        }

    };

    inline void Warning(const std::string& msg)
    {

        if (itsDebugState <= kWarningMsg)
        {
            printf("Warning::%s %s\n", itsUserName.c_str(), msg.c_str());
        }

    };

    inline void Error(const std::string& msg)
    {
        if (itsDebugState <= kErrorMsg)
        {
            printf("Error::%s %s\n", itsUserName.c_str(), msg.c_str());
        }

    };

    inline void Fatal(const std::string& msg)
    {

        if (itsDebugState <= kFatalMsg)
        {
            printf("Fatal::%s %s\n", itsUserName.c_str(), msg.c_str());
        }

    };

private:
    HPDebugState itsDebugState;
    std::string itsUserName;

};

} // namespace himan

#endif /* LOGGER_H */
