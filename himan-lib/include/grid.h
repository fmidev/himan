/**
 * @file grid.h
 *
 */

#ifndef GRID_H
#define GRID_H

/**
 * @class grid
 *
 * @brief Interface for all grids
 */

#include "earth_shape.h"
#include "himan_common.h"
#include "logger.h"
#include "matrix.h"
#include "point.h"
#include "serialization.h"
#include <map>
#include <vector>

class OGRPolygon;

#ifdef HAVE_CEREAL
#include <ogr_spatialref.h>

namespace cereal
{
template <class Archive>
inline std::string save_minimal(const Archive& ar, const OGRSpatialReference& sp)
{
	char* projstr;
	if (sp.exportToWkt(&projstr) != OGRERR_NONE)
	{
		throw std::runtime_error("Failed to get WKT");
	}

	std::string proj(projstr);
	CPLFree(projstr);

	return proj;
}

template <class Archive>
inline void load_minimal(const Archive& ar, OGRSpatialReference& sp, const std::string& str)
{
	if (sp.importFromWkt(str.c_str()) != OGRERR_NONE)
	{
		throw std::runtime_error("Failed to import from WKT");
	}
}
}  // namespace cereal
#else
class OGRSpatialReference;
class OGRCoordinateTransformation;
#endif

namespace himan
{
enum HPGridClass
{
	kUnknownGridClass = 0,
	kRegularGrid,
	kIrregularGrid
};

const std::unordered_map<HPGridClass, std::string> HPGridClassToString = {
    {kUnknownGridClass, "unknown"}, {kRegularGrid, "regular"}, {kIrregularGrid, "irregular"}};

enum HPGridType
{
	kUnknownGridType = 0,
	kLatitudeLongitude = 1,
	kStereographic,
	kAzimuthalEquidistant,
	kRotatedLatitudeLongitude,
	kLambertConformalConic,
	kReducedGaussian,
	kLambertEqualArea,
	kTransverseMercator,
	kPointList = 100
};

const std::unordered_map<HPGridType, std::string> HPGridTypeToString = {{kUnknownGridType, "unknown grid type"},
                                                                        {kLatitudeLongitude, "ll"},
                                                                        {kStereographic, "polster"},
                                                                        {kAzimuthalEquidistant, "azimuthal"},
                                                                        {kRotatedLatitudeLongitude, "rll"},
                                                                        {kLambertConformalConic, "lcc"},
                                                                        {kReducedGaussian, "rgg"},
                                                                        {kLambertEqualArea, "laea"},
                                                                        {kTransverseMercator, "tm"},
                                                                        {kPointList, "pointlist"}};

enum HPScanningMode
{
	kUnknownScanningMode = 0,
	kTopLeft = 17,      // +x-y
	kTopRight = 18,     // -x-y
	kBottomLeft = 33,   // +x+y
	kBottomRight = 34,  // -x+y
};

const std::unordered_map<std::string, HPScanningMode> HPScanningModeFromString = {{"unknown", kUnknownScanningMode},
                                                                                  {"+x-y", kTopLeft},
                                                                                  {"-x-y", kTopRight},
                                                                                  {"+x+y", kBottomLeft},
                                                                                  {"-x+y", kBottomRight}};

const std::unordered_map<HPScanningMode, std::string> HPScanningModeToString = {{kUnknownScanningMode, "unknown"},
                                                                                {kTopLeft, "+x-y"},
                                                                                {kTopRight, "-x-y"},
                                                                                {kBottomLeft, "+x+y"},
                                                                                {kBottomRight, "-x+y"}};

class grid
{
   public:
	grid() = delete;
	virtual ~grid() = default;

	grid(const grid&) = default;
	grid& operator=(const grid&) = default;

	virtual std::string ClassName() const
	{
		return "himan::grid";
	}
	virtual bool operator==(const grid& other) const;
	virtual bool operator!=(const grid& other) const;

	virtual std::ostream& Write(std::ostream& file) const;

	virtual std::unique_ptr<grid> Clone() const = 0;

	/*
	 * Functions that are common and valid to all types of grids,
	 * and are implemented in this class.
	 */

	HPGridType Type() const;
	HPGridClass Class() const;

	/*
	 * Functions that are common and valid to all types of grids.
	 *
	 * For those functions that clearly have some kind of default
	 * implementation, that implementation is done in grid-class,
	 * but so that it can be overridden in inheriting classes.
	 *
	 * Functions whos implementation depends on the grid type are
	 * declared abstract should be implemented by deriving classes.
	 */

	virtual size_t Size() const;

	virtual point FirstPoint() const = 0;
	virtual point LastPoint() const = 0;

	/* Return latlon coordinates of a given grid point */
	virtual point LatLon(size_t locationIndex) const = 0;

	/* Return a unique key */
	virtual size_t Hash() const = 0;

	bool UVRelativeToGrid() const;
	void UVRelativeToGrid(bool theUVRelativeToGrid);

	virtual earth_shape<double> EarthShape() const = 0;

	/**
	 * @brief Create a list of point locations for a grid in the grids native coordinate system
	 *
	 */

	virtual std::vector<point> GridPointsInProjectionSpace() const;

	std::string Name() const;

   protected:
	grid(HPGridClass gridClass, HPGridType gridType, bool itsUVRelativeToGrid, const std::string& theName = "");

	bool EqualsTo(const grid& other) const;

	HPGridClass itsGridClass;
	HPGridType itsGridType;

	logger itsLogger;

	/**
	 * True if parameter UV components are grid relative, false if they are earth-relative.
	 * This has  no meaning for:
	 * - parameters what are not vector components
	 * - non-projected data
	 */

	bool itsUVRelativeToGrid;
	std::string itsName;

#ifdef HAVE_CEREAL

	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsGridClass), CEREAL_NVP(itsGridType), CEREAL_NVP(itsUVRelativeToGrid));
	}

	template <class Archive>
	static void load_and_construct(Archive& ar, cereal::construct<grid>& construct)
	{
		HPGridClass c;
		HPGridType t;
		bool uv;
		ar(c, t, uv);
		construct(c, t, uv);
	}

#endif
};

class regular_grid : public grid
{
   public:
	regular_grid() = delete;
	~regular_grid() = default;
	regular_grid(const regular_grid&);
	regular_grid& operator=(const regular_grid& other) = delete;

	virtual std::ostream& Write(std::ostream& file) const override;

	/* Return grid point value (incl. fractions) of a given latlon point */
	virtual point XY(const point& latlon) const;

	/* Return grid point value (location) of a given target grid */
	virtual std::vector<point> XY(const regular_grid& to) const;

	/* Return latitude-longitude value for a given location index */
	virtual point LatLon(size_t i) const override;

	/* Return latitude-longitude value for a given point in projected space */
	virtual point LatLon(const point& projected) const;

	/* Return projected value for a given latitude-longitude point */
	virtual point Projected(const point& latlon) const;

	/*
	 * Functions that are only valid for some grid types, but for ease
	 * of use they are declared here. It is up to the actual grid classes
	 * to implement correct functionality.
	 */

	virtual point BottomLeft() const;
	virtual point TopRight() const;
	virtual point TopLeft() const;
	virtual point BottomRight() const;
	virtual point FirstPoint() const override;
	virtual point LastPoint() const override;

	virtual HPScanningMode ScanningMode() const;

	virtual size_t Size() const override;
	virtual size_t Ni() const;
	virtual size_t Nj() const;
	virtual double Di() const;
	virtual double Dj() const;

	virtual std::string Proj4String() const;
	virtual std::string WKT(const std::map<std::string, std::string>& opts = {}) const;
	virtual std::unique_ptr<OGRPolygon> Geometry() const;
	virtual earth_shape<double> EarthShape() const override;

	virtual std::unique_ptr<OGRSpatialReference> SpatialReference() const;
	virtual std::vector<point> GridPointsInProjectionSpace() const override;

   protected:
	regular_grid(HPGridType gridType, HPScanningMode scMode, double di, double dj, size_t ni, size_t nj,
	             bool uvRelativeToGrid, const std::string& theName = "");

	bool EqualsTo(const regular_grid& other) const;

	HPScanningMode itsScanningMode;
	std::unique_ptr<OGRSpatialReference> itsSpatialReference;
	std::unique_ptr<OGRCoordinateTransformation> itsXYToLatLonTransformer;
	std::unique_ptr<OGRCoordinateTransformation> itsLatLonToXYTransformer;

	double itsDi;
	double itsDj;
	size_t itsNi;
	size_t itsNj;

#ifdef HAVE_CEREAL

	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsScanningMode), CEREAL_NVP(itsDi), CEREAL_NVP(itsDj), CEREAL_NVP(itsNi), CEREAL_NVP(itsNj));
	}

	template <class Archive>
	static void load_and_construct(Archive& ar, cereal::construct<regular_grid>& construct)
	{
		HPScanningMode sm;
		double di, dj;
		size_t ni, nj;
		ar(sm, di, dj, ni, nj);
		construct(sm, di, dj, ni, nj);
	}
#endif
};

class irregular_grid : public grid
{
   public:
	irregular_grid(HPGridType type);
	~irregular_grid() = default;
	irregular_grid(const irregular_grid&);
	irregular_grid& operator=(const irregular_grid& other) = delete;
	virtual earth_shape<double> EarthShape() const override;
	virtual void EarthShape(const earth_shape<double>& theShape);

   protected:
	bool EqualsTo(const irregular_grid& other) const;
	earth_shape<double> itsEarthShape;

#ifdef HAVE_CEREAL
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(cereal::base_class<grid>(this));
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const grid& ob)
{
	return ob.Write(file);
}

inline std::ostream& operator<<(std::ostream& file, const regular_grid& ob)
{
	return ob.Write(file);
}

}  // namespace himan

// Specialize the fmt::formatter template for HPScanningMode
// needed because of fmt v11 and grid_cache
template <>
struct fmt::formatter<himan::HPScanningMode>
{
	template <typename ParseContext>
	constexpr auto parse(ParseContext& ctx)
	{
		return ctx.begin();
	}

	template <typename FormatContext>
	auto format(const himan::HPScanningMode& c, FormatContext& ctx) const -> decltype(ctx.out())
	{
		return fmt::format_to(ctx.out(), "{}", fmt::underlying(c));
	}
};

#ifdef HAVE_CEREAL
CEREAL_REGISTER_TYPE(himan::regular_grid);
CEREAL_REGISTER_POLYMORPHIC_RELATION(himan::grid, himan::regular_grid);
#endif

#endif /* GRID_H */
