/*
 * timer_factory.cpp
 *
 */

#include "timer_factory.h"

using namespace himan;

std::unique_ptr<timer_factory> timer_factory::itsInstance;

timer_factory* timer_factory::Instance()
{
	if (!itsInstance)
	{
		itsInstance = std::unique_ptr<timer_factory>(new timer_factory());
	}

	return itsInstance.get();
}

timer* timer_factory::GetTimer() { return new timer(); }
