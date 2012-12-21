/*
 * logger_factory.cpp
 *
 *  Created on: Dec 17, 2012
 *      Author: partio
 */

#include "logger_factory.h"
using namespace hilpee;

logger_factory* logger_factory::itsInstance = NULL;

logger_factory::logger_factory() : itsDebugStateMain(kInfoMsg)
{

}

logger_factory::~logger_factory()
{
	if (itsInstance)
	{
		delete itsInstance;
	}
}

logger_factory* logger_factory::Instance()
{
	if (!itsInstance)
	{
		itsInstance = new logger_factory();
	}

	return itsInstance;
}
std::unique_ptr<logger> logger_factory::GetLog(const std::string& theUserName)
{
	return std::unique_ptr<logger> (new logger(theUserName, itsDebugStateMain));
}

void logger_factory::DebugState(HPDebugState theDebugState)
{
	itsDebugStateMain = theDebugState;
}

HPDebugState logger_factory::DebugState()
{
	return itsDebugStateMain;
}


