/**
 *  * @class rotation
 *  *
 *  * @brief Define a rotation object that contains coefficients of a quaternion.
 *  * With rotate functions this rotation can be applied to a position object to perform coordinate rotation.
 *  * Principles of quaternion rotation can be found from https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
 *  * In particular the following concepts are used:
 *  *   a) p' = q*p*q^-1 for rotation of position vector p
 *  *   b) q' = q1*q2 for combining a chain of rotations into one object
 *  * Methods are implemented through Eigen library geometry module
 * https://eigen.tuxfamily.org/dox/group__TutorialGeometry.html
 *  */

#include "position.h"

#ifndef ROTATION_H
#define ROTATION_H

namespace himan
{
namespace geoutil
{
template <typename T>
class rotation
{
   public:
	T* Data()
	{
		return itsValues;
	}
	const T* Data() const
	{
		return itsValues;
	}

	// create a rotation object that transforms coordinates from rotated LatLon to regular LatLon
	static rotation<T> FromRotLatLon(const T& latOfSouthPole, const T& lonOfSouthPole, const T& angleOfRotation);

	// create a rotation object that transforms coordinates from regular LatLon to rotated LatLon
	static rotation<T> ToRotLatLon(const T& latOfSouthPole, const T& lonOfSouthPole, const T& angleOfRotation);

   private:
	T itsValues[4];
};

// rotate a position
template <typename T>
void rotate(position<T>&, const rotation<T>&);

// create a rotated copy of a position
template <typename T>
position<T> rotate(const position<T>&, const rotation<T>&);

}  // end namespace geoutil
}  // end namespace himan
#endif /* POSITION_H */
