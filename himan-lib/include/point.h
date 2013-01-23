/**
 * @file point.h
 *
 * @date Jan 16, 2013
 * @author partio
 */

#ifndef POINT_H
#define POINT_H

#include <ostream>
#include <NFmiPoint.h>

namespace himan
{

/**
 * @class point
 *
 * @brief Define simple XY point. Mimics NFmiPoint in some aspects,
 * but functionality is re-written to avoid including newbase headers.
 */

class point
{
public:
	point();
	point(double itsX, double itsY);

	/**
	 * @brief Constructor to integrate himan::point to newbase. If/when we don't use newbase
	 * for interpolation, this can be removed.
	 */

    point(const NFmiPoint& thePoint);
	~point() {}

	point(const point& other);
	point& operator=(const point& other);

	bool operator==(const point& thePoint) const;
	bool operator!=(const point& thePoint) const;

	std::string ClassName() const
	{
		return "himan::point";
	}

	HPVersionNumber Version() const
	{
		return HPVersionNumber(0, 1);
	}

	/**
	 * @return X coordinate value
	 */

	double X() const;

	/**
	 * @return Y coordinate value
	 */

	double Y() const;

	void X(double theX);
	void Y(double theY);

    std::ostream& Write(std::ostream& file) const
    {
		file << "<" << ClassName() << " " << Version() << ">" << std::endl;
		file << "__itsX__ " << itsX << std::endl;
		file << "__itsY__ " << itsY << std::endl;

		return file;
    }

    /**
     * @brief Create NFmiPoint from himan::point. If/when we don't use newbase
     * for interpolation, this can be removed.
     */

    NFmiPoint ToNFmiPoint() const;


private:
	double itsX;
	double itsY;
};

inline point::point() : itsX(kHPMissingFloat), itsY(kHPMissingFloat)
{
}

inline point::point(double theX, double theY) : itsX(theX), itsY(theY)
{
}

inline point::point(const point& other) : itsX(other.X()), itsY(other.Y())
{
}

inline point::point(const NFmiPoint& other) : itsX(other.X()), itsY(other.Y())
{
}

inline point& point::operator=(const point& other)
{
	itsX = other.X();
	itsY = other.Y();

	return *this;
}

inline bool point::operator==(const point& other) const
{
	const double kCoordinateEpsilon = 0.00001;

	bool yEquals = (fabs(itsY - other.Y()) < kCoordinateEpsilon);
	bool xEquals = (fabs(itsX - other.X()) < kCoordinateEpsilon);

	return (xEquals && yEquals);
}

inline bool point::operator!=(const point& thePoint) const
{
	return !(*this == thePoint);
}

inline double point::X() const
{
	return itsX;
}

inline double point::Y() const
{
	return itsY;
}

inline void point::X(double theX)
{
	itsX = theX;
}

inline void point::Y(double theY)
{
	itsY = theY;
}

inline NFmiPoint point::ToNFmiPoint() const
{
	return NFmiPoint(itsX, itsY);
}

inline std::ostream& operator<<(std::ostream& file, const point& ob)
{
    return ob.Write(file);
}

} // namespace himan

#endif /* POINT_H */
