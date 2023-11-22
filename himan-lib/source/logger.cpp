/*
 * logger.cpp
 *
 */

#include "logger.h"

namespace himan
{
HPDebugState logger::MainDebugState = himan::kDebugMsg;

logger::logger() : itsDebugState(MainDebugState), itsUserName("HimanDefaultLogger")
{
}
logger::logger(const std::string& theUserName) : itsDebugState(MainDebugState), itsUserName(theUserName)
{
}
logger::logger(const std::string& theUserName, HPDebugState theDebugState)
    : itsDebugState(theDebugState), itsUserName(theUserName)
{
}

}  // namespace himan
