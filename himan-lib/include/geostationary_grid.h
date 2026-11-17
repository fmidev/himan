/**
 * @file   geostationary_grid.h
 *
 * Geostationary satellite grid ("space view perspective": GRIB2 grid definition
 * template 3.90, GRIB1 data representation type 90), as used for example by
 * MTG FCI and MSG SEVIRI. Himan reads and writes it only as GRIB2.
 *
 * The grid is regular in the projection space of a geostationary satellite
 * (proj: +proj=geos), i.e. grid points are evenly spaced in scan angle.
 * Parts of the grid (including all corners for a full disk image) are not on
 * the earth disk: for such points LatLon() returns a missing point.
 *
 * The grid is defined with the GRIB native parameters which are stored as-is so
 * that writing the grid back to grib reproduces section 3 exactly.
 *
 * Projected coordinates follow the regular_grid convention: the first grid point
 * is at projected coordinates (0,0). The "true" geostationary projection
 * coordinates (origin at sub-satellite point) are available with
 * ProjectedFirstPoint() and GeosToLatLon() / LatLonToGeos().
 */

#ifndef GEOSTATIONARY_GRID_H
#define GEOSTATIONARY_GRID_H

#include "grid.h"
#include "logger.h"
#include "point.h"
#include "serialization.h"
#include <string>

namespace himan
{
class geostationary_grid : public regular_grid
{
   public:
	/**
	 * @param ni Number of points along x axis (Nx)
	 * @param nj Number of points along y axis (Ny)
	 * @param nr Altitude of the camera from earth center, in units of earth equatorial radius * 10^6 (Nr)
	 * @param dx Apparent diameter of earth in grid lengths, x direction (dx)
	 * @param dy Apparent diameter of earth in grid lengths, y direction (dy)
	 * @param xp X coordinate of sub-satellite point, in units of 10^-3 grid lengths (Xp)
	 * @param yp Y coordinate of sub-satellite point, in units of 10^-3 grid lengths (Yp)
	 * @param subSatelliteLongitude Longitude of sub-satellite point, in units of 10^-6 degrees
	 * @param subSatelliteLatitude Latitude of sub-satellite point, in units of 10^-6 degrees (must be zero)
	 * @param sweepAxis Sweep angle axis of the instrument, 'x' or 'y'. GRIB does not encode this.
	 *                  EUMETSAT (MSG, MTG) uses 'y', GOES uses 'x'.
	 */
	geostationary_grid(HPScanningMode theScanningMode, size_t ni, size_t nj, long nr, long dx, long dy, long xp,
	                   long yp, long subSatelliteLongitude, long subSatelliteLatitude,
	                   const earth_shape<double>& earthShape, char sweepAxis = 'y', const std::string& theName = "");

	virtual ~geostationary_grid() = default;
	geostationary_grid(const geostationary_grid& other);
	geostationary_grid& operator=(const geostationary_grid& other) = delete;

	virtual std::string ClassName() const override
	{
		return "himan::geostationary_grid";
	}
	virtual std::ostream& Write(std::ostream& file) const override;

	std::unique_ptr<grid> Clone() const override;
	size_t Hash() const override;

	bool operator==(const grid& other) const override;
	bool operator!=(const grid& other) const override;

	/* Closed-form geostationary projection (same formulas as PROJ geos); these do not use OGR.
	 * Points that are not on the earth disk are returned as missing.
	 *
	 * Why not the inherited OGR path: a full disk has ~22% of points off the earth, and the
	 * per-point OGR transform fails (and logs an error) for each of them; per-point OGR calls are
	 * slow for 10^8 points; and the closed form only reads constants so it is thread safe without
	 * locking. This keeps LatLon() cheap enough that no lat/lon cache is needed (a cache would be
	 * ~1 GB per 1 km grid). Verified against ecCodes and OGR in unit_geostationary_grid. */

	point LatLon(size_t locationIndex) const override;
	point LatLon(const point& projected) const override;
	point Projected(const point& latlon) const override;

	std::vector<point> XY(const regular_grid& target) const override;
	using regular_grid::XY;

	std::vector<point> GridPointsInProjectionSpace() const override;

	/* Geostationary projection coordinates (metres, origin at sub-satellite point) of the first grid point */
	point ProjectedFirstPoint() const;

	/* Convert geos coordinates (origin at sub-satellite point) to latlon and vice versa */
	point GeosToLatLon(double x, double y) const;
	point LatLonToGeos(const point& latlon) const;

	long Nr() const;
	long Dx() const;
	long Dy() const;
	long Xp() const;
	long Yp() const;
	long SubSatelliteLongitude() const;
	long SubSatelliteLatitude() const;
	char SweepAxis() const;

	/* Satellite height above earth surface (at equator) in metres */
	double Height() const;

	/* Sub-satellite point longitude in degrees */
	double Orientation() const;

	/* Geostationary grids are only defined for scanning mode +x-y */
	static bool IsSupportedScanningMode(HPScanningMode mode);

   private:
	bool EqualsTo(const geostationary_grid& other) const;
	bool IsSameProjection(const geostationary_grid& other) const;
	void Init(const earth_shape<double>& earthShape);

	long itsNr;
	long itsDx;
	long itsDy;
	long itsXp;
	long itsYp;
	long itsSubSatelliteLongitude;
	long itsSubSatelliteLatitude;
	char itsSweepAxis;

	// Derived values

	double itsA;              // semi-major axis
	double itsB;              // semi-minor axis
	double itsH;              // satellite height above surface
	double itsLon0;           // sub-satellite longitude, degrees
	double itsFalseEasting;   // xp * di
	double itsFalseNorthing;  // -yp * dj

#ifdef HAVE_CEREAL
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar) const
	{
		ar(CEREAL_NVP(itsScanningMode), CEREAL_NVP(itsNi), CEREAL_NVP(itsNj), CEREAL_NVP(itsNr), CEREAL_NVP(itsDx),
		   CEREAL_NVP(itsDy), CEREAL_NVP(itsXp), CEREAL_NVP(itsYp), CEREAL_NVP(itsSubSatelliteLongitude),
		   CEREAL_NVP(itsSubSatelliteLatitude), CEREAL_NVP(itsA), CEREAL_NVP(itsB), CEREAL_NVP(itsSweepAxis),
		   CEREAL_NVP(itsName));
	}

	template <class Archive>
	static void load_and_construct(Archive& ar, cereal::construct<geostationary_grid>& construct)
	{
		HPScanningMode sm;
		size_t ni, nj;
		long nr, dx, dy, xp, yp, lon, lat;
		double a, b;
		char sweep;
		std::string name;

		ar(sm, ni, nj, nr, dx, dy, xp, yp, lon, lat, a, b, sweep, name);
		construct(sm, ni, nj, nr, dx, dy, xp, yp, lon, lat, earth_shape<double>(a, b), sweep, name);
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const geostationary_grid& ob)
{
	return ob.Write(file);
}
}  // namespace himan

#ifdef HAVE_CEREAL
CEREAL_REGISTER_TYPE(himan::geostationary_grid);
CEREAL_REGISTER_POLYMORPHIC_RELATION(himan::regular_grid, himan::geostationary_grid);
#endif

#endif /* GEOSTATIONARY_GRID_H */
