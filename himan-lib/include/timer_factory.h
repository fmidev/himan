/*
 * timer_factory.h
 *
 *  Created on: Dec 17, 2012
 *      Author: partio
 */

#ifndef TIMER_FACTORY_H
#define TIMER_FACTORY_H

#include "timer.h"
#include <memory>

namespace himan
{
class timer_factory
{
   public:
	static timer_factory* Instance();

	timer_factory(const timer_factory& other) = delete;
	timer_factory& operator=(const timer_factory& other) = delete;

	timer* GetTimer();

	~timer_factory() = default;

   private:
	timer_factory() {}
	/*#ifdef __CUDACC__
	    timer_factory(const timer_factory& other);
	    timer_factory& operator=(const timer_factory& other);
	#endif
	*/
	static std::unique_ptr<timer_factory> itsInstance;
};

}  // namespace himan

#endif /* TIMER_FACTORY_H */
