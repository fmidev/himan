/**
 * @class position
 *
 * @brief Define position vector from origin to a point in 3d space.
 * Coordinate transformations are taken from:
 * Datums and Map Projections 
 * for Remote Sensing, GIS and Surveying 
 * 2nd Edition
 * ISBN 978-1-904445-47-0
 *
 */

#include "earth_shape.h"
#include <cmath>

#ifndef POSITION_H
#define POSITION_H

template <typename T>
class position
{
   public:
	position()
	{
	}
	// From cartesian coordinates
	position(const T& theX, const T& theY, const T& theZ) : itsValues({theX, theY, theZ})
	{
	}
	// From Lat, Lon, H on an ellipsoid 
	// Eq. 2.6 - 2.8 and Appendix C.2
	position(const T& theLat, const T& theLon, const T& theHeight, const himan::earth_shape<T>& theShape)
	{
		T v = theShape.A() / std::sqrt(1.0 - theShape.E2() * std::sin(theLat) * std::sin(theLat));
		itsValues[0] = (v + theHeight) * std::cos(theLat) * std::cos(theLon);
		itsValues[1] = (v + theHeight) * std::cos(theLat) * std::sin(theLon);
		itsValues[2] = (v * (1 - theShape.E2()) + theHeight) * std::sin(theLat);
	}

	T X() const
	{
		return itsValues[0];
	}
	T Y() const
	{
		return itsValues[1];
	}
	T Z() const
	{
		return itsValues[2];
	}

	T Lat(const himan::earth_shape<T>&) const;
	T Lon(const himan::earth_shape<T>&) const;
	T H(const himan::earth_shape<T>&) const;

	T* Data()
	{
		return itsValues;
	}
	const T* Data() const
	{
		return itsValues;
	}

   private:
	T itsValues[3];
};

// Operators
// Comparison
template <typename T>
bool operator==(const position<T>& p1, const position<T>& p2)
{
	return (p1.X() == p2.X() && p1.Y() == p2.Y() && p1.Z() == p2.Z());
}

// Addition
template <typename T>
position<T> operator+(const position<T>& p1, const position<T>& p2)
{
	return position<T>(p1.X() + p2.X(), p1.Y() + p2.Y(), p1.Z() + p2.Z());
}

// Subtraction
template <typename T>
position<T> operator-(const position<T>& p1, const position<T>& p2)
{
	return position<T>(p1.X() - p2.X(), p1.Y() - p2.Y(), p1.Z() - p2.Z());
}

// Scaling
template <typename T>
position<T> operator*(const T& t, const position<T>& p)
{
	return position<T>(t * p.X(), t * p.Y(), t * p.Z());
}

template <typename T>
position<T> operator*(const position<T>& p, const T& t)
{
	return t * p;
}

// Longitude from Eq. 2.9
template <typename T>
T position<T>::Lon(const himan::earth_shape<T>& theShape) const
{
	const T x = itsValues[0];
	const T y = itsValues[1];
	return std::atan(y / x);
}

// Latitude from Eq. 2.10 - 2.14
template <typename T>
T position<T>::Lat(const himan::earth_shape<T>& theShape) const
{
	const T x = itsValues[0];
	const T y = itsValues[1];
	const T z = itsValues[2];

	const T p = std::sqrt(x * x + y * y);
	const T u = std::atan(z / p * theShape.A() / theShape.B());
	const T epsilon = theShape.E2() / (1.0 - theShape.E2());
	return std::atan((z + (epsilon * theShape.B() * std::pow(std::sin(u), 3))) /
	                 (p - theShape.E2() * theShape.A() * std::pow(std::cos(u), 3)));
}

// Height obove ellipsoid surface 2.11 - 2.14
template <typename T>
T position<T>::H(const himan::earth_shape<T>& theShape) const
{
	const T x = itsValues[0];
	const T y = itsValues[1];

	const T v = theShape.A() / std::sqrt(1.0 - theShape.E2() * Lat(theShape) * Lat(theShape));
	const T p = std::sqrt(x * x + y * y);
	return p * 1.0 / std::cos(Lat(theShape)) - v;
}

#endif /* POSITION_H */
