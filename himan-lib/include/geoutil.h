/**
 * @file geoutil.h
 *
 *
 * @brief Utility namespace for geographic helper functions and classes
 */

#include "point.h"

#ifndef GEOUTIL_H
#define GEOUTIL_H

namespace himan
{
/**
 * @namespace himan::util
 * @brief Namespace for all utility functions and classes
 **/

namespace geoutil
{
double Distance(const himan::point& a, const himan::point& b, double r = 1.0);

// area of a spherical triangle with corner points P1|P2|P3
double Area(const himan::point& P1, const himan::point& P2, const himan::point& P3, double r = 1.0);

}  // namespace geoutil
}  // namespace himan

#endif /* GEOUTIL_H */
