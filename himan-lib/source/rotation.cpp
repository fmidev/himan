#include "rotation.h"
#include "himan_common.h"
#include <Eigen/Geometry>

using namespace Eigen;

template <typename T>
void himan::geoutil::rotate(himan::geoutil::position<T>& p, const himan::geoutil::rotation<T>& r)
{
	// Map data structures to Eigen library objects
	Map<Matrix<T, 3, 1>> P(p.Data());
	Map<const Quaternion<T>> QR(r.Data());

	// Create a corresponding quaternion of the input position vector
	Quaternion<T> QP;
	QP.w() = 0;
	QP.vec() = P;

	// Apply spatial rotation through quaternion products
	Quaternion<T> rotatedP = QR * QP * QR.inverse();
	P = rotatedP.vec();
}
template void himan::geoutil::rotate<float>(himan::geoutil::position<float>&, const himan::geoutil::rotation<float>&);
template void himan::geoutil::rotate<double>(himan::geoutil::position<double>&,
                                             const himan::geoutil::rotation<double>&);

template <typename T>
himan::geoutil::position<T> himan::geoutil::rotate(const himan::geoutil::position<T>& p,
                                                   const himan::geoutil::rotation<T>& r)
{
	position<T> ret(p);
	rotate(ret, r);
	return ret;
}
template himan::geoutil::position<float> himan::geoutil::rotate<float>(const himan::geoutil::position<float>&,
                                                                       const himan::geoutil::rotation<float>&);
template himan::geoutil::position<double> himan::geoutil::rotate<double>(const himan::geoutil::position<double>&,
                                                                         const himan::geoutil::rotation<double>&);

template <typename T>
himan::geoutil::rotation<T> himan::geoutil::rotation<T>::FromRotLatLon(const T& latOfSouthPole, const T& lonOfSouthPole,
                                                                       const T& angleOfRot)
{
	himan::geoutil::rotation<T> ret;

	// Map data structures to Eigen library objects
	Map<Quaternion<T>> QRot(ret.Data());

	// Create a rotation quaternion from product of a series of rotations about principle axis
	QRot = AngleAxis<T>(lonOfSouthPole, Matrix<T, 3, 1>::UnitZ()) *
	       AngleAxis<T>(-(T(M_PI / 2.0) + latOfSouthPole), Matrix<T, 3, 1>::UnitY()) *
	       AngleAxis<T>(-angleOfRot, Matrix<T, 3, 1>::UnitZ());

	return ret;
}
template himan::geoutil::rotation<float> himan::geoutil::rotation<float>::FromRotLatLon(const float& latOfSouthPole,
                                                                                        const float& lonOfSouthPole,
                                                                                        const float& angleOfRot);
template himan::geoutil::rotation<double> himan::geoutil::rotation<double>::FromRotLatLon(const double& latOfSouthPole,
                                                                                          const double& lonOfSouthPole,
                                                                                          const double& angleOfRot);

template <typename T>
himan::geoutil::rotation<T> himan::geoutil::rotation<T>::ToRotLatLon(const T& latOfSouthPole, const T& lonOfSouthPole,
                                                                     const T& angleOfRot)
{
	himan::geoutil::rotation<T> ret;
	// Map data structures to Eigen library objects
	Map<Quaternion<T>> QRot(ret.Data());

	// Create a rotation quaternion from product of a series of rotations about principle axis
	QRot = AngleAxis<T>(angleOfRot, Matrix<T, 3, 1>::UnitZ()) *
	       AngleAxis<T>(T(M_PI / 2.0) + latOfSouthPole, Matrix<T, 3, 1>::UnitY()) *
	       AngleAxis<T>(-lonOfSouthPole, Matrix<T, 3, 1>::UnitZ());

	return ret;
}
template himan::geoutil::rotation<float> himan::geoutil::rotation<float>::ToRotLatLon(const float& latOfSouthPole,
                                                                                      const float& lonOfSouthPole,
                                                                                      const float& angleOfRot);
template himan::geoutil::rotation<double> himan::geoutil::rotation<double>::ToRotLatLon(const double& latOfSouthPole,
                                                                                        const double& lonOfSouthPole,
                                                                                        const double& angleOfRot);
