/**
 * @file   lambert_conformal_grid.h
 *
 */

#ifndef LAMBERT_CONFORMAL_GRID_H
#define LAMBERT_CONFORMAL_GRID_H

#include "grid.h"
#include "logger.h"
#include "point.h"
#include "serialization.h"
#include <string>

class OGRCoordinateTransformation;
class OGRSpatialReference;

namespace himan
{
class lambert_conformal_grid : public regular_grid
{
   public:
	lambert_conformal_grid();
	lambert_conformal_grid(HPScanningMode theScanningMode, point theFirstPoint);

	virtual ~lambert_conformal_grid();
	lambert_conformal_grid(const lambert_conformal_grid& other);
	lambert_conformal_grid& operator=(const lambert_conformal_grid& other) = delete;

	virtual std::string ClassName() const
	{
		return "himan::lambert_conformal_grid";
	}
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
	 * @return Distance between two points in X axis in meters
	 */

	double Di() const override;

	/**
	 * @return Distance between two points in Y axis in meters
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
	void TopLeft(const point& theTopLeft);

	point FirstPoint() const;
	point LastPoint() const;

	bool operator==(const grid& other) const;
	bool operator!=(const grid& other) const;

	point XY(const point& latlon) const override;
	point LatLon(size_t locationIndex) const override;

	std::unique_ptr<grid> Clone() const override;

	void Orientation(double theOrientation);
	double Orientation() const;

	void StandardParallel1(double theStandardParallel1);
	double StandardParallel1() const;

	void StandardParallel2(double theStandardParallel2);
	double StandardParallel2() const;

	OGRSpatialReference SpatialReference() const;

	point SouthPole() const;
	void SouthPole(const point& theSouthPole);

	size_t Hash() const override;

	double Cone() const;

   private:
	bool EqualsTo(const lambert_conformal_grid& other) const;
	void SetCoordinates() const;

	point itsBottomLeft;
	point itsTopLeft;

	double itsDi;
	double itsDj;

	size_t itsNi;
	size_t itsNj;

	double itsOrientation;
	double itsStandardParallel1;
	mutable double itsStandardParallel2;

	point itsSouthPole;

	mutable std::once_flag itsAreaFlag;
	mutable std::unique_ptr<OGRCoordinateTransformation> itsXYToLatLonTransformer;
	mutable std::unique_ptr<OGRCoordinateTransformation> itsLatLonToXYTransformer;
	mutable std::unique_ptr<OGRSpatialReference> itsSpatialReference;
#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(cereal::base_class<regular_grid>(this), CEREAL_NVP(itsBottomLeft), CEREAL_NVP(itsTopLeft), CEREAL_NVP(itsDi),
		   CEREAL_NVP(itsDj), CEREAL_NVP(itsNi), CEREAL_NVP(itsNj), CEREAL_NVP(itsOrientation),
		   CEREAL_NVP(itsStandardParallel1), CEREAL_NVP(itsStandardParallel2), CEREAL_NVP(itsSouthPole));
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const lambert_conformal_grid& ob)
{
	return ob.Write(file);
}
}  // namespace himan

#ifdef SERIALIZATION
CEREAL_REGISTER_TYPE(himan::lambert_conformal_grid);
#endif

#endif /* LAMBERT_CONFORMAL_GRID_H */
