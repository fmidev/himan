/*
 * timer_factory.cpp
 *
 *  Created on: Dec 17, 2012
 *      Author: partio
 */

#include "timer_factory.h"

using namespace himan;

std::unique_ptr<timer_factory> timer_factory::itsInstance = NULL;

timer_factory* timer_factory::Instance()
{
    if (!itsInstance)
    {
        itsInstance = std::unique_ptr<timer_factory> (new timer_factory());
    }

    return itsInstance.get();
}

timer* timer_factory::GetTimer()
{
    return new timer();
}
