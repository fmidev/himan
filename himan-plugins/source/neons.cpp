/*
 * neons.cpp
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 */

#include "neons.h"
#include "logger_factory.h"

using namespace himan::plugin;

neons::neons() : itsNeonsDB(std::shared_ptr<NFmiNeonsDB> (new NFmiNeonsDB()))
{
	itsLogger = logger_factory::Instance()->GetLog("neons");
}


std::shared_ptr<NFmiNeonsDB> neons::Neons()
{
	return itsNeonsDB;
}
