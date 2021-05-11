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

namespace himan
{
class lambert_conformal_grid : public regular_grid
{
   public:
	lambert_conformal_grid(HPScanningMode theScanningMode, const point& theFirstPoint, size_t ni, size_t nj, double di,
	                       double dj, double theOrientation, double theStandardParallel1, double theStandardParallel2,
	                       const earth_shape<double>& earthShape, bool firstPointIsProjected = false,
	                       const std::string& theName = "");
	lambert_conformal_grid(HPScanningMode theScanningMode, const point& theFirstPoint, size_t ni, size_t nj, double di,
	                       double dj, std::unique_ptr<OGRSpatialReference> spRef, bool firstPointIsProjected = false,
	                       const std::string& theName = "");

	virtual ~lambert_conformal_grid() = default;
	lambert_conformal_grid(const lambert_conformal_grid& other);
	lambert_conformal_grid& operator=(const lambert_conformal_grid& other) = delete;

	virtual std::string ClassName() const override
	{
		return "himan::lambert_conformal_grid";
	}
	virtual std::ostream& Write(std::ostream& file) const;

	bool operator==(const grid& other) const;
	bool operator!=(const grid& other) const;

	std::unique_ptr<grid> Clone() const override;

	double Orientation() const;
	double StandardParallel1() const;
	double StandardParallel2() const;

	size_t Hash() const override;
	double Cone() const;

   private:
	bool EqualsTo(const lambert_conformal_grid& other) const;
	void CreateCoordinateTransformations(const point& firstPoint, bool isProjected);

#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar) const
	{
		ar(CEREAL_NVP(itsScanningMode), CEREAL_NVP(FirstPoint()), CEREAL_NVP(itsNi), CEREAL_NVP(itsNj),
		   CEREAL_NVP(itsDi), CEREAL_NVP(itsDj), CEREAL_NVP(itsSpatialReference));
	}

	template <class Archive>
	static void load_and_construct(Archive& ar, cereal::construct<lambert_conformal_grid>& construct)
	{
		HPScanningMode sm;
		point fp;
		size_t ni, nj;
		double di, dj;
		std::unique_ptr<OGRSpatialReference> sp;

		ar(sm, fp, ni, nj, di, dj, sp);
		construct(sm, fp, ni, nj, di, dj, std::move(sp));
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
CEREAL_REGISTER_POLYMORPHIC_RELATION(himan::regular_grid, himan::lambert_conformal_grid);
#endif

#endif /* LAMBERT_CONFORMAL_GRID_H */
