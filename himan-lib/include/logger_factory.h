/*
 * logger_factory.h
 *
 *  Created on: Dec 17, 2012
 *      Author: partio
 */

#ifndef LOGGER_FACTORY_H
#define LOGGER_FACTORY_H

#include "logger.h"
#include "himan_common.h"

namespace himan
{

class logger_factory
{

	public:
		static logger_factory* Instance();

		std::unique_ptr<logger> GetLog(const std::string& theUserName);

		void DebugState(HPDebugState theDebugState);
		HPDebugState DebugState();

	private:
		logger_factory();
		~logger_factory();
		logger_factory(const logger_factory& other) = delete;
		logger_factory& operator=(const logger_factory& other) = delete;

		HPDebugState itsDebugStateMain;
		static logger_factory* itsInstance;
};

} // namespace himan

#endif /* LOGGER_FACTORY_H */
