/**
 * @file logger_factory.h
 *
 */

#ifndef LOGGER_FACTORY_H
#define LOGGER_FACTORY_H

#include "himan_common.h"
#include "logger.h"  // Include logger.h here since logger_factory.h is used *everywhere*
                     // and by including logger.h we need only include one file. And besides
                     // logger.h isn't *that* big!

namespace himan
{
class logger;

class logger_factory
{
   public:
	static logger_factory* Instance();

	logger_factory(const logger_factory& other) = delete;
	~logger_factory() = default;
	logger_factory& operator=(const logger_factory& other) = delete;

	std::unique_ptr<logger> GetLog(const std::string& theUserName);

	void DebugState(HPDebugState theDebugState);
	HPDebugState DebugState();

   private:
	logger_factory();

	HPDebugState itsDebugStateMain;
	static std::unique_ptr<logger_factory> itsInstance;
};

}  // namespace himan

#endif /* LOGGER_FACTORY_H */
