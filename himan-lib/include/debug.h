/**
 * @file debug.h
 *
 */

#ifndef HIMAN_DEBUG_H_
#define HIMAN_DEBUG_H

#define HIMAN_DEBUG_BREAK __asm__ __volatile__ ("int $3")

#ifdef DEBUG
#  ifndef __CUDACC__
#    define ASSERT(Expr)                                                           \
    do {                                                                           \
        if (!(Expr)) {                                                             \
            if (himan::AssertionFailed(#Expr, __LINE__, __FUNCTION__, __FILE__)) { \
               HIMAN_DEBUG_BREAK;                                                  \
            } else {                                                               \
               himan::Abort();                                                     \
            }                                                                      \
        }                                                                          \
    } while (false)
#  else
// TODO: proper CUDA version?
#include <cassert>
#define ASSERT(Expr) assert((Expr))
#  endif
#else
#define ASSERT(Expr)
#endif

namespace himan
{
/// @brief Prints out a description of the error and returns true if we're running inside a debugger.
bool AssertionFailed(const char* expr, long line, const char* fn, const char* file);
/// @brief Abort execution of the program, and print out a stacktrace.
void Abort() __attribute__((noreturn));
} // namespace himan

// HIMAN_DEBUG_H
#endif
