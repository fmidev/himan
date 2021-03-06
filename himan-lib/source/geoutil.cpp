#include "geoutil.h"
#include "position.h"
#include <Eigen/Dense>
#include <cmath>

using namespace himan;
using namespace Eigen;

double geoutil::Distance(const point& a, const point& b, double r)
{
	const double alpha = std::pow(std::sin((b.Y() - a.Y()) / 360 * M_PI), 2) +
	               std::cos(a.Y() / 180 * M_PI) * std::cos(b.Y() / 180 * M_PI) *
	                   std::pow(std::sin((b.X() - a.X()) / 360 * M_PI), 2);
	return r * 2 * std::atan2(std::sqrt(alpha), std::sqrt(1 - alpha));
}

double geoutil::Area(const point& P1, const point& P2, const point& P3, double r)
{
	const position<double> p1(P1.Y() / 180 * M_PI, P1.Y() / 180 * M_PI, 0, earth_shape<double>(r));
	const position<double> p2(P2.Y() / 180 * M_PI, P2.X() / 180 * M_PI, 0, earth_shape<double>(r));
	const position<double> p3(P3.Y() / 180 * M_PI, P3.X() / 180 * M_PI, 0, earth_shape<double>(r));

	const Matrix<double, 3, 1> A(p1.Data());
	const Matrix<double, 3, 1> B(p2.Data());
	const Matrix<double, 3, 1> C(p3.Data());
	const double a = std::atan(A.cross(B).norm() / A.dot(B));
	const double b = std::atan(B.cross(C).norm() / B.dot(C));
	const double c = std::atan(C.cross(A).norm() / C.dot(A));
	const double s = (a + b + c) / 2;
	return 4 *
	       std::atan(
	           std::sqrt(std::tan(s / 2) * std::tan((s - a) / 2) * std::tan((s - b) / 2) * std::tan((s - c) / 2))) *
	       r * r;
}

double geoutil::Bearing(const point& a, const point& b)
{
	return std::atan2(std::sin((b.X() - a.X())/180*M_PI) * std::cos(b.Y()/180*M_PI) , std::cos(a.Y()/180*M_PI) * std::sin(b.Y()/180*M_PI) - std::sin(a.Y()/180*M_PI) * std::cos(b.Y()/180*M_PI) * std::cos((b.X() - a.X())/180*M_PI));
}

bool geoutil::InsideTriangle(const point& a, const point& b, const point& c, const point& p)
{
	const double epsilon = 1e-12;

	position<double> A(a.Y() / 180 * M_PI, a.X() / 180 * M_PI, 0, earth_shape<double>(1));
	position<double> B(b.Y() / 180 * M_PI, b.X() / 180 * M_PI, 0, earth_shape<double>(1));
	position<double> C(c.Y() / 180 * M_PI, c.X() / 180 * M_PI, 0, earth_shape<double>(1));

	position<double> P(p.Y() / 180 * M_PI, p.X() / 180 * M_PI, 0, earth_shape<double>(1));

	Matrix<double, 3, 1> v(P.Data());
	Matrix<double, 3, 3> M;
	M << A.X(), B.X(), C.X(), A.Y(), B.Y(), C.Y(), A.Z(), B.Z(), C.Z();

	ColPivHouseholderQR<Matrix<double, 3, 3>> dec(M);

	Matrix<double, 3, 1> beta = dec.solve(v);

	return beta(0) >= -epsilon && beta(1) >= -epsilon && beta(2) >= -epsilon;
}
