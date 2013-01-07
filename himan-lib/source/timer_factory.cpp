/*
 * timer_factory.cpp
 *
 *  Created on: Dec 17, 2012
 *      Author: partio
 */

#include "timer_factory.h"

using namespace himan;

timer_factory* timer_factory::itsInstance = NULL;

timer_factory::~timer_factory()
{
    if (itsInstance)
    {
        delete itsInstance;
    }
}

timer_factory* timer_factory::Instance()
{
    if (!itsInstance)
    {
        itsInstance = new timer_factory();
    }

    return itsInstance;
}

timer* timer_factory::GetTimer()
{
    return new timer();
}
