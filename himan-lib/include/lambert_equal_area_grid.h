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

	virtual ~lambert_equal_area_grid() = default;
	lambert_equal_area_grid(const lambert_equal_area_grid& other);
	lambert_equal_area_grid& operator=(const lambert_equal_area_grid& other) = delete;

	virtual std::string ClassName() const override
	{
		return "himan::lambert_equal_area_grid";
	}
	virtual std::ostream& Write(std::ostream& file) const;

	std::unique_ptr<grid> Clone() const override;
	size_t Hash() const override;

	bool operator==(const grid& other) const;
	bool operator!=(const grid& other) const;

	double Orientation() const;
	double StandardParallel() const;
	OGRSpatialReference SpatialReference() const;

   private:
	bool EqualsTo(const lambert_equal_area_grid& other) const;
	void CreateCoordinateTransformations(const point& firstPoint, bool isProjected);

#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(cereal::base_class<regular_grid>(this));
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
