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
	                         bool firstPointIsProjected = false, const std::string& theName = "");

	transverse_mercator_grid(HPScanningMode theScanningMode, const point& theFirstPoint, size_t ni, size_t nj,
	                         double di, double dj, std::unique_ptr<OGRSpatialReference> spRef,
	                         bool firstPointIsProjected = false, const std::string& theName = "");

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

   private:
	bool EqualsTo(const transverse_mercator_grid& other) const;
	void CreateCoordinateTransformations(const point& firstPoint, bool isProjected);

#ifdef HAVE_CEREAL
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar) const
	{
		ar(CEREAL_NVP(itsScanningMode), CEREAL_NVP(FirstPoint()), CEREAL_NVP(itsNi), CEREAL_NVP(itsNj),
		   CEREAL_NVP(itsDi), CEREAL_NVP(itsDj), CEREAL_NVP(itsSpatialReference));
	}

	template <class Archive>
	static void load_and_construct(Archive& ar, cereal::construct<transverse_mercator_grid>& construct)
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

inline std::ostream& operator<<(std::ostream& file, const transverse_mercator_grid& ob)
{
	return ob.Write(file);
}
}  // namespace himan

#ifdef HAVE_CEREAL
CEREAL_REGISTER_TYPE(himan::transverse_mercator_grid);
CEREAL_REGISTER_POLYMORPHIC_RELATION(himan::regular_grid, himan::transverse_mercator_grid);
#endif

#endif /* TRANSVERSE_MERCATOR_H */
