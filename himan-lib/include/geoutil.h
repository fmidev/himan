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
// check if a point p is inside of a triangle formed by a|b|c
bool InsideTriangle(const himan::point& a, const himan::point& b, const himan::point& c, const himan::point& p);

// spherical distance from point a to b
double Distance(const himan::point& a, const himan::point& b, double r = 1.0);

// area of a spherical triangle with corner points P1|P2|P3
double Area(const himan::point& P1, const himan::point& P2, const himan::point& P3, double r = 1.0);

}  // namespace geoutil
}  // namespace himan

#endif /* GEOUTIL_H */
