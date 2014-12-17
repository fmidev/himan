/**
 * @file point.h
 *
 * @date Jan 16, 2013
 * @author partio
 */

#ifndef POINT_H
#define POINT_H

#include <ostream>
#include "himan_common.h"

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
	~point() {}

	point(const point& other);
	point& operator=(const point& other);

	bool operator==(const point& thePoint) const;
	bool operator!=(const point& thePoint) const;

	std::string ClassName() const
	{
		return "himan::point";
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
		file << "<" << ClassName() << ">" << std::endl;
		file << "__itsX__ " << itsX << std::endl;
		file << "__itsY__ " << itsY << std::endl;

		return file;
	}

private:
	double itsX;
	double itsY;
};

inline point::point() : itsX(kHPMissingValue), itsY(kHPMissingValue)
{
}

inline point::point(double theX, double theY) : itsX(theX), itsY(theY)
{
}

inline point::point(const point& other) : itsX(other.X()), itsY(other.Y())
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

inline std::ostream& operator<<(std::ostream& file, const point& ob)
{
	return ob.Write(file);
}

class station : public point
{
public:
	station();
	station(int theId, const std::string& theName, double lon, double lat);

	int Id() const;
	void Id(int theId);
	
	std::string Name() const;
	void Name(const std::string& theName);
	
private:
	int itsId; // FMISID
	std::string itsName;
};

inline station::station()
	: point()
	, itsId(kHPMissingInt)
	, itsName("Himan default station")
{}

inline station::station(int theId, const std::string& theName, double lon, double lat)
	: point(lon,lat)
	, itsId(theId)
	, itsName(theName)
{}

inline int station::Id() const
{
	return itsId;
}

inline void station::Id(int theId)
{
	itsId = theId;
}

inline std::string station::Name() const
{
	return itsName;
}

inline void station::Name(const std::string& theName)
{
	itsName = theName;
}


} // namespace himan

#endif /* POINT_H */
