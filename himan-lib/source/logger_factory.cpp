/**
 * @file logger_factory.cpp
 *
 */

#include "logger_factory.h"

using namespace himan;

std::unique_ptr<logger_factory> logger_factory::itsInstance;

logger_factory::logger_factory() : itsDebugStateMain(kInfoMsg) {}
logger_factory* logger_factory::Instance()
{
	if (!itsInstance)
	{
		itsInstance = std::unique_ptr<logger_factory>(new logger_factory());
	}

	return itsInstance.get();
}

std::unique_ptr<logger> logger_factory::GetLog(const std::string& theUserName)
{
	return std::unique_ptr<logger>(new logger(theUserName, itsDebugStateMain));  // no make_unique in C++11 :(
}

void logger_factory::DebugState(HPDebugState theDebugState) { itsDebugStateMain = theDebugState; }
HPDebugState logger_factory::DebugState() { return itsDebugStateMain; }
