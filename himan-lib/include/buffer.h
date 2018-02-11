#pragma once
#include <stdlib.h>

namespace himan
{
struct buffer
{
        unsigned char* data;
        unsigned long length;
        buffer() : data(0), length(0)
        {
        }
        ~buffer()
        {
                if (data)
                {
                        free(data);
                }
        }
};
} // namespace himan
