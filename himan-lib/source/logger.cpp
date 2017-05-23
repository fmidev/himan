/*
 * logger.cpp
 *
 */

#include "logger.h"

himan::logger::logger() : itsDebugState(kInfoMsg), itsUserName("HimanDefaultLogger") {}
himan::logger::logger(const std::string& theUserName, HPDebugState theDebugState)
    : itsDebugState(theDebugState), itsUserName(theUserName)
{
}