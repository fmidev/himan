/**
 * @file   lambert_equal_area_grid.h
 *
 */

#ifndef LAMBERT_EQUAL_AREA_H
#define LAMBERT_EQUAL_AREA_H

#include "grid.h"
#include "logger.h"
#include "point.h"
#include "serialization.h"
#include <string>

class OGRCoordinateTransformation;
class OGRSpatialReference;

namespace himan
{
class lambert_equal_area_grid : public regular_grid
{
   public:
	lambert_equal_area_grid(HPScanningMode theScanningMode, const point& theFirstPoint, size_t ni, size_t nj, double di,
	                        double dj, double theOrientation, double theStandardParallel,
	                        const earth_shape<double>& earthShape, bool firstPointIsProjected = false);
	lambert_equal_area_grid(HPScanningMode theScanningMode, const point& theFirstPoint, size_t ni, size_t nj, double di,
	                        double dj, std::unique_ptr<OGRSpatialReference> spRef, bool firstPointIsProjected = false);

	virtual ~lambert_equal_area_grid();
	lambert_equal_area_grid(const lambert_equal_area_grid& other);
	lambert_equal_area_grid& operator=(const lambert_equal_area_grid& other) = delete;

	virtual std::string ClassName() const
	{
		return "himan::lambert_equal_area_grid";
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

	point FirstPoint() const override;
	point LastPoint() const override;

	std::unique_ptr<grid> Clone() const override;
	size_t Hash() const override;

	point XY(const point& latlon) const override;
	point LatLon(size_t locationIndex) const override;

	bool operator==(const grid& other) const;
	bool operator!=(const grid& other) const;

	double Orientation() const;
	double StandardParallel() const;
	OGRSpatialReference SpatialReference() const;

   private:
	bool EqualsTo(const lambert_equal_area_grid& other) const;
	void CreateCoordinateTransformations(const point& firstPoint, bool isProjected);

	double itsDi;
	double itsDj;

	size_t itsNi;
	size_t itsNj;

	mutable std::unique_ptr<OGRCoordinateTransformation> itsXYToLatLonTransformer;
	mutable std::unique_ptr<OGRCoordinateTransformation> itsLatLonToXYTransformer;
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

inline std::ostream& operator<<(std::ostream& file, const lambert_equal_area_grid& ob)
{
	return ob.Write(file);
}
}  // namespace himan

#ifdef SERIALIZATION
CEREAL_REGISTER_TYPE(himan::lambert_equal_area_grid);
#endif

#endif /* LAMBERT_EQUAL_AREA_H */
