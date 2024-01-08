#pragma once

#include "cuda_helper.h"
#include "himan_common.h"
#include "serialization.h"
#include <boost/functional/hash.hpp>
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
	CUDA_HOST CUDA_DEVICE ~point()
	{
	}
	CUDA_HOST CUDA_DEVICE point(const point& other);
	CUDA_HOST CUDA_DEVICE point& operator=(const point& other);

	CUDA_HOST CUDA_DEVICE bool operator==(const point& thePoint) const;
	CUDA_HOST CUDA_DEVICE bool operator!=(const point& thePoint) const;
	CUDA_HOST operator std::string() const;

	std::string ClassName() const
	{
		return "himan::point";
	}
	CUDA_HOST CUDA_DEVICE double X() const;
	CUDA_HOST CUDA_DEVICE double Y() const;

	CUDA_HOST CUDA_DEVICE void X(double theX);
	CUDA_HOST CUDA_DEVICE void Y(double theY);

#ifndef __CUDACC__
	size_t Hash() const;
#endif

	CUDA_HOST CUDA_DEVICE static bool LatLonCompare(const point& a, const point& b);
	std::ostream& Write(std::ostream& file) const;

   protected:
	double itsX;
	double itsY;

#ifdef HAVE_CEREAL
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsX), CEREAL_NVP(itsY));
	}
#endif
};

CUDA_HOST CUDA_DEVICE inline point::point() : itsX(kHPMissingValue), itsY(kHPMissingValue)
{
}
CUDA_HOST CUDA_DEVICE inline point::point(double theX, double theY) : itsX(theX), itsY(theY)
{
}
CUDA_HOST CUDA_DEVICE inline point::point(const point& other) : itsX(other.X()), itsY(other.Y())
{
}
CUDA_HOST CUDA_DEVICE inline point& point::operator=(const point& other)
{
	itsX = other.X();
	itsY = other.Y();

	return *this;
}

CUDA_HOST CUDA_DEVICE inline bool point::operator==(const point& other) const
{
	const bool yEquals = (fabs(itsY - other.itsY) < kCoordinateEpsilon);
	const bool xEquals = (fabs(itsX - other.itsX) < kCoordinateEpsilon);
	return (xEquals && yEquals);
}

CUDA_HOST CUDA_DEVICE inline bool point::LatLonCompare(const point& a, const point& b)
{
	const bool yEquals = (fabs(a.Y() - b.Y()) < kCoordinateEpsilon);
	const bool xEquals =
	    (fabs(a.X() - b.X()) < kCoordinateEpsilon) ||
	    (a.X() > 180. && (fabs((a.X() - 360.) - b.X())) < kCoordinateEpsilon) ||  // "a" has longitude > 360
	    (b.X() > 180. && (fabs(a.X() - (b.X() - 360.))) < kCoordinateEpsilon);    // "b" has longitude > 360
	return (xEquals && yEquals);
}

CUDA_HOST CUDA_DEVICE inline bool point::operator!=(const point& thePoint) const
{
	return !(*this == thePoint);
}
CUDA_HOST CUDA_DEVICE inline double point::X() const
{
	return itsX;
}
CUDA_HOST CUDA_DEVICE inline double point::Y() const
{
	return itsY;
}
CUDA_HOST CUDA_DEVICE inline void point::X(double theX)
{
	itsX = theX;
}
CUDA_HOST CUDA_DEVICE inline void point::Y(double theY)
{
	itsY = theY;
}
#ifndef __CUDACC__
inline size_t point::Hash() const
{
	size_t ret = 0;
	boost::hash_combine(ret, std::hash<double>{}(itsX));
	boost::hash_combine(ret, std::hash<double>{}(itsY));
	return ret;
}
#endif
inline std::ostream& operator<<(std::ostream& file, const point& ob)
{
	return ob.Write(file);
}
CUDA_HOST inline point::operator std::string() const
{
	return std::to_string(itsX) + "," + std::to_string(itsY);
}

}  // namespace himan
