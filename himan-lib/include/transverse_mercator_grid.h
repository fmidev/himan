/**
 * @file   transverse_mercator_grid.h
 *
 */

#ifndef TRANSVERSE_MERCATOR_H
#define TRANSVERSE_MERCATOR_H

#include "grid.h"
#include "logger.h"
#include "point.h"
#include "serialization.h"
#include <string>

namespace himan
{
class transverse_mercator_grid : public regular_grid
{
   public:
	transverse_mercator_grid(HPScanningMode theScanningMode, const point& theFirstPoint, size_t ni, size_t nj,
	                         double di, double dj, double orientation, double theStandardParallel, double theScale,
	                         double falseEasting, double falseNorthing, const earth_shape<double>& earthShape,
	                         bool firstPointIsProjected = false);

	transverse_mercator_grid(HPScanningMode theScanningMode, const point& theFirstPoint, size_t ni, size_t nj,
	                         double di, double dj, std::unique_ptr<OGRSpatialReference> spRef,
	                         bool firstPointIsProjected = false);

	virtual ~transverse_mercator_grid() = default;
	transverse_mercator_grid(const transverse_mercator_grid& other);
	transverse_mercator_grid& operator=(const transverse_mercator_grid& other) = delete;

	virtual std::string ClassName() const override
	{
		return "himan::transverse_mercator_grid";
	}
	virtual std::ostream& Write(std::ostream& file) const;

	std::unique_ptr<grid> Clone() const override;
	size_t Hash() const override;

	bool operator==(const grid& other) const;
	bool operator!=(const grid& other) const;

	double Orientation() const;
	double StandardParallel() const;
	double Scale() const;

	OGRSpatialReference SpatialReference() const;

   private:
	bool EqualsTo(const transverse_mercator_grid& other) const;
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

inline std::ostream& operator<<(std::ostream& file, const transverse_mercator_grid& ob)
{
	return ob.Write(file);
}
}  // namespace himan

#ifdef SERIALIZATION
CEREAL_REGISTER_TYPE(himan::transverse_mercator_grid);
#endif

#endif /* TRANSVERSE_MERCATOR_H */
