/*
 * File:   stereographic_grid.h
 *
 */

#ifndef STEREOGRAPHIC_GRID_H
#define STEREOGRAPHIC_GRID_H

#include "grid.h"
#include "logger.h"
#include "serialization.h"
#include <string>

#if defined __GNUC__ && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ <= 6))
#define override  // override specifier not support until 4.8
#endif

namespace himan
{
class stereographic_grid : public regular_grid
{
   public:
	stereographic_grid(HPScanningMode theScanningMode, const point& theFirstPoint, size_t ni, size_t nj, double di,
	                   double dj, double theOrientation, double latin, double lat_ts,
	                   const earth_shape<double>& earthShape, bool firstPointIsProjected = false,
	                   const std::string& theName = "");
	stereographic_grid(HPScanningMode theScanningMode, const point& theFirstPoint, size_t ni, size_t nj, double di,
	                   double dj, double theOrientation, const earth_shape<double>& earthShape,
	                   bool firstPointIsProjected = false, const std::string& theName = "");
	stereographic_grid(HPScanningMode theScanningMode, const point& theFirstPoint, size_t ni, size_t nj, double di,
	                   double dj, std::unique_ptr<OGRSpatialReference> spRef, bool firstPointIsProjected = false,
	                   const std::string& theName = "");

	virtual ~stereographic_grid() = default;
	/**
	 * @brief Copy constructor for stereographic_grid
	 *
	 * When stereographic_grid is copied, the contents (ie. class d_matrix_t) is copied as
	 * well.
	 *
	 * @param other
	 */

	stereographic_grid(const stereographic_grid& other);
	stereographic_grid& operator=(const stereographic_grid& other) = delete;

	std::string ClassName() const override
	{
		return "himan::stereographic_grid";
	}
	std::ostream& Write(std::ostream& file) const override;

	double Orientation() const;
	double LatitudeOfCenter() const;
	double LatitudeOfOrigin() const;

	bool operator==(const grid& other) const;
	bool operator!=(const grid& other) const;

	size_t Hash() const override;

	std::unique_ptr<grid> Clone() const override;

   private:
	void CreateCoordinateTransformations(const point& firstPoint, bool isProjected);
	bool EqualsTo(const stereographic_grid& other) const;

#ifdef HAVE_CEREAL
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& archive) const
	{
		archive(CEREAL_NVP(itsScanningMode), CEREAL_NVP(FirstPoint()), CEREAL_NVP(itsNi), CEREAL_NVP(itsNj),
		        CEREAL_NVP(itsDi), CEREAL_NVP(itsDj), CEREAL_NVP(itsSpatialReference));
	}

	template <class Archive>
	static void load_and_construct(Archive& ar, cereal::construct<stereographic_grid>& construct)
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

inline std::ostream& operator<<(std::ostream& file, const stereographic_grid& ob)
{
	return ob.Write(file);
}
}  // namespace himan

#ifdef HAVE_CEREAL
CEREAL_REGISTER_TYPE(himan::stereographic_grid);
CEREAL_REGISTER_POLYMORPHIC_RELATION(himan::regular_grid, himan::stereographic_grid);
#endif

#endif /* STEREOGRAPHIC_GRID_H */
