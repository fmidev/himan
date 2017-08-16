/*
 * File:   latitude_longitude_grid.h
 *
 */

#ifndef LATITUDE_LONGITUDE_GRID_H
#define LATITUDE_LONGITUDE_GRID_H

#include "grid.h"
#include "logger.h"
#include "packed_data.h"
#include "point.h"
#include "serialization.h"
#include <string>

class NFmiRotatedLatLonArea;
namespace himan
{
class latitude_longitude_grid : public grid
{
   public:
	latitude_longitude_grid();
	latitude_longitude_grid(HPScanningMode theScanningMode, point theBottomLeft, point theTopRight);

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

	virtual std::string ClassName() const { return "himan::latitude_longitude_grid"; }
	virtual std::ostream& Write(std::ostream& file) const;

	/**
	 * @return Number of points along X axis
	 */

	size_t Ni() const override;

	/**
	 * @return Number of points along Y axis
	 */

	size_t Nj() const override;

	/**
	 *
	 * @return  Grid size
	 */

	size_t Size() const override;

	/**
	 * @return Distance between two points in X axis in degrees
	 */

	double Di() const override;

	/**
	 * @return Distance between two points in Y axis in degrees
	 */

	double Dj() const override;

	void Ni(size_t theNi);
	void Nj(size_t theNj);

	void Di(double theDi);
	void Dj(double theDj);

	point BottomLeft() const override;
	point TopRight() const override;
	point BottomRight() const;
	point TopLeft() const;

	void BottomLeft(const point& theBottomLeft);
	void TopRight(const point& theTopRight);
	void BottomRight(const point& theBottomRight);
	void TopLeft(const point& theTopLeft);

	void FirstPoint(const point& theFirstPoint);
	void LastPoint(const point& theLastPoint);

	point FirstPoint() const;
	point LastPoint() const;

	bool operator==(const grid& other) const;
	bool operator!=(const grid& other) const;

	void PackedData(std::unique_ptr<packed_data> thePackedData);
	packed_data& PackedData();

	bool Swap(HPScanningMode newScanningMode) override;

	point XY(const point& latlon) const override;
	point LatLon(size_t locationIndex) const override;

	latitude_longitude_grid* Clone() const override;

   protected:
	void UpdateCoordinates() const;
	bool EqualsTo(const latitude_longitude_grid& other) const;

	mutable point itsBottomLeft;
	mutable point itsTopRight;
	mutable point itsBottomRight;
	mutable point itsTopLeft;

	mutable double itsDi;
	mutable double itsDj;

	size_t itsNi;
	size_t itsNj;

   private:
	mutable bool itsIsGlobal;

#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(cereal::base_class<grid>(this), CEREAL_NVP(itsBottomLeft), CEREAL_NVP(itsBottomRight),
		   CEREAL_NVP(itsTopLeft), CEREAL_NVP(itsTopRight), CEREAL_NVP(itsDi), CEREAL_NVP(itsDj), CEREAL_NVP(itsNi),
		   CEREAL_NVP(itsNj), CEREAL_NVP(itsIsGlobal));
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const latitude_longitude_grid& ob) { return ob.Write(file); }
class rotated_latitude_longitude_grid : public latitude_longitude_grid
{
   public:
	rotated_latitude_longitude_grid();
	rotated_latitude_longitude_grid(HPScanningMode theScanningMode, point theBottomLeft, point theTopRight,
	                                point theSouthPole, bool initiallyRotated = true);

	virtual ~rotated_latitude_longitude_grid();
	rotated_latitude_longitude_grid(const rotated_latitude_longitude_grid& other);
	rotated_latitude_longitude_grid& operator=(const rotated_latitude_longitude_grid& other) = delete;

	bool operator==(const grid& other) const;
	bool operator!=(const grid& other) const;

	virtual std::ostream& Write(std::ostream& file) const;

	virtual std::string ClassName() const { return "himan::rotated_latitude_longitude_grid"; }
	rotated_latitude_longitude_grid* Clone() const override;

	point SouthPole() const;
	void SouthPole(const point& theSouthPole);

	point XY(const point& latlon) const override;
	point LatLon(size_t locationIndex) const override;
	point RotatedLatLon(size_t locationIndex) const;

   private:
	bool EqualsTo(const rotated_latitude_longitude_grid& other) const;
	mutable std::unique_ptr<NFmiRotatedLatLonArea> itsRotLatLonArea;

	point itsSouthPole;

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
