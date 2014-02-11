/*
 * logger.cpp
 *
 *  Created on: Nov 21, 2012
 *      Author: partio
 */

#include "logger.h"

himan::logger::logger()
	: itsDebugState(kInfoMsg)
	, itsUserName("HimanDefaultLogger")
{}

himan::logger::logger(const std::string& theUserName, HPDebugState theDebugState) 
	: itsDebugState(theDebugState)
	, itsUserName(theUserName)
{}

