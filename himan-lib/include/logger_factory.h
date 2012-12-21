/*
 * hilpee_logger_factory.h
 *
 *  Created on: Dec 17, 2012
 *      Author: partio
 */

#ifndef LOGGER_FACTORY_H
#define LOGGER_FACTORY_H

#include "logger.h"
#include "hilpee_common.h"

namespace hilpee
{

class logger_factory
{

	public:
		static logger_factory* Instance();

#ifdef HAVE_CPP11
		std::unique_ptr<logger> GetLog(const std::string& theUserName);
#else
		std::auto_ptr<logger> GetLog(const std::string& theUserName);
#endif

		void DebugState(HPDebugState theDebugState);
		HPDebugState DebugState();

	private:
		logger_factory();
		~logger_factory();
		logger_factory(const logger_factory&) {}

		HPDebugState itsDebugStateMain;
		static logger_factory* itsInstance;
};

} // namespace hilpee

#endif /* LOGGER_FACTORY_H */
