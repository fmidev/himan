/*
 * File:   latitude_longitude_grid.h
 *
 */

#ifndef LATITUDE_LONGITUDE_GRID_H
#define LATITUDE_LONGITUDE_GRID_H

#include "grid.h"
#include "logger.h"
#include "point.h"
#include "rotation.h"
#include "serialization.h"
#include <mutex>
#include <string>

class NFmiRotatedLatLonArea;
namespace himan
{
class latitude_longitude_grid : public regular_grid
{
   public:
	latitude_longitude_grid(HPScanningMode theScanningMode, const point& theFirstPoint, const point& theLastPoint,
	                        size_t ni, size_t nj, const earth_shape<double>& earthShape);
	latitude_longitude_grid(HPScanningMode theScanningMode, const point& theFirstPoint, size_t ni, size_t nj, double di,
	                        double dj, const earth_shape<double>& earthShape);

	virtual ~latitude_longitude_grid() = default;
	/**
	 * @brief Copy constructor for latitude_longitude_grid
	 *
	 * When latitude_longitude_grid is copied, the contents (ie. class d_matrix_t) is copied as
	 * well.
	 *
	 * @param other
	 */

	latitude_longitude_grid(const latitude_longitude_grid& other);
	latitude_longitude_grid& operator=(const latitude_longitude_grid& other) = delete;

	virtual std::string ClassName() const override
	{
		return "himan::latitude_longitude_grid";
	}
	virtual std::ostream& Write(std::ostream& file) const;

	virtual point FirstPoint() const override;
	bool IsGlobal() const;

	bool operator==(const grid& other) const;
	bool operator!=(const grid& other) const;

	point XY(const point& latlon) const override;
	point LatLon(size_t locationIndex) const override;

	size_t Hash() const override;

	std::unique_ptr<grid> Clone() const override;

	virtual std::string Proj4String() const override;
	virtual earth_shape<double> EarthShape() const override;

   protected:
	bool EqualsTo(const latitude_longitude_grid& other) const;

	point itsFirstPoint;
	earth_shape<double> itsEarthShape;

   private:
#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(cereal::base_class<regular_grid>(this), CEREAL_NVP(itsFirstPoint), CEREAL_NVP(itsEarthShape)

		);
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const latitude_longitude_grid& ob)
{
	return ob.Write(file);
}
class rotated_latitude_longitude_grid : public latitude_longitude_grid
{
   public:
	rotated_latitude_longitude_grid(HPScanningMode theScanningMode, const point& theFirstPoint, size_t ni, size_t nj,
	                                double di, double dj, const earth_shape<double>& earthShape,
	                                const point& theSouthPole, bool initiallyRotated = true);
	rotated_latitude_longitude_grid(HPScanningMode theScanningMode, const point& theFirstPoint,
	                                const point& theLastPoint, size_t ni, size_t nj,
	                                const earth_shape<double>& earthShape, const point& theSouthPole,
	                                bool initiallyRotated = true);

	virtual ~rotated_latitude_longitude_grid() = default;
	rotated_latitude_longitude_grid(const rotated_latitude_longitude_grid& other);
	rotated_latitude_longitude_grid& operator=(const rotated_latitude_longitude_grid& other) = delete;

	bool operator==(const grid& other) const;
	bool operator!=(const grid& other) const;

	virtual std::ostream& Write(std::ostream& file) const;

	virtual std::string ClassName() const override
	{
		return "himan::rotated_latitude_longitude_grid";
	}
	std::unique_ptr<grid> Clone() const override;

	// return south pole location in normal latlon
	point SouthPole() const;

	// return first point in normal latlon
	point FirstPoint() const override;
	// return grid xy coordinates for normal latlon
	point XY(const point& latlon) const override;
	// return latlon for grid running index
	point LatLon(size_t locationIndex) const override;
	// return rotated point for grid running index
	point RotatedLatLon(size_t locationIndex) const;
	// return rotated point for normal latlon
	point Rotate(const point& latlon) const;

	size_t Hash() const override;

	virtual std::string Proj4String() const override;

   private:
	bool EqualsTo(const rotated_latitude_longitude_grid& other) const;
	point itsSouthPole;

	himan::geoutil::rotation<double> itsFromRotLatLon;
	himan::geoutil::rotation<double> itsToRotLatLon;

#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(cereal::base_class<latitude_longitude_grid>(this), CEREAL_NVP(itsSouthPole));
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const rotated_latitude_longitude_grid& ob)
{
	return ob.Write(file);
}
}  // namespace himan

#ifdef SERIALIZATION
CEREAL_REGISTER_TYPE(himan::latitude_longitude_grid);
CEREAL_REGISTER_TYPE(himan::rotated_latitude_longitude_grid);
#endif
#endif /* LATITUDE_LONGITUDE_GRID_H */
