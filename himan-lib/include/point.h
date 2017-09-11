#pragma once

#include "cuda_helper.h"
#include "himan_common.h"
#include "serialization.h"
#include <ostream>

namespace himan
{
/**
 * @class point
 *
 * @brief Define simple XY point. Mimics NFmiPoint in some aspects,
 * but functionality is re-written to avoid including newbase headers.
 */

const double kCoordinateEpsilon = 0.00001;

class point
{
   public:
	CUDA_HOST CUDA_DEVICE point();
	CUDA_HOST CUDA_DEVICE point(double itsX, double itsY);
	CUDA_HOST CUDA_DEVICE ~point() {}
	CUDA_HOST CUDA_DEVICE point(const point& other);
	CUDA_HOST CUDA_DEVICE point& operator=(const point& other);

	CUDA_HOST CUDA_DEVICE bool operator==(const point& thePoint) const;
	CUDA_HOST CUDA_DEVICE bool operator!=(const point& thePoint) const;

	std::string ClassName() const { return "himan::point"; }
	CUDA_HOST CUDA_DEVICE double X() const;
	CUDA_HOST CUDA_DEVICE double Y() const;

	CUDA_HOST CUDA_DEVICE void X(double theX);
	CUDA_HOST CUDA_DEVICE void Y(double theY);

	std::ostream& Write(std::ostream& file) const;

   protected:
	double itsX;
	double itsY;

#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsX), CEREAL_NVP(itsY));
	}
#endif
};

CUDA_HOST CUDA_DEVICE inline point::point() : itsX(kHPMissingValue), itsY(kHPMissingValue) {}
CUDA_HOST CUDA_DEVICE inline point::point(double theX, double theY) : itsX(theX), itsY(theY) {}
CUDA_HOST CUDA_DEVICE inline point::point(const point& other) : itsX(other.X()), itsY(other.Y()) {}
CUDA_HOST CUDA_DEVICE inline point& point::operator=(const point& other)
{
	itsX = other.X();
	itsY = other.Y();

	return *this;
}

CUDA_HOST CUDA_DEVICE inline bool point::operator==(const point& other) const
{
	bool yEquals = (fabs(itsY - other.Y()) < kCoordinateEpsilon);
	bool xEquals = (fabs(itsX - other.X()) < kCoordinateEpsilon);

	return (xEquals && yEquals);
}

CUDA_HOST CUDA_DEVICE inline bool point::operator!=(const point& thePoint) const { return !(*this == thePoint); }
CUDA_HOST CUDA_DEVICE inline double point::X() const { return itsX; }
CUDA_HOST CUDA_DEVICE inline double point::Y() const { return itsY; }
CUDA_HOST CUDA_DEVICE inline void point::X(double theX) { itsX = theX; }
CUDA_HOST CUDA_DEVICE inline void point::Y(double theY) { itsY = theY; }
inline std::ostream& operator<<(std::ostream& file, const point& ob) { return ob.Write(file); }
}  // namespace himan
