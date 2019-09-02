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

namespace himan
{
class grid
{
   public:
	grid();
	grid(const std::string& WKT);

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
	void Type(HPGridType theGridType);

	HPGridClass Class() const;
	void Class(HPGridClass theGridClass);

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

	earth_shape<double> EarthShape() const;
	void EarthShape(const earth_shape<double>& theEarthShape);

	std::string WKT() const;
	std::string Proj4() const;

   protected:
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

	earth_shape<double> itsEarthShape;

#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsGridClass), CEREAL_NVP(itsGridType), CEREAL_NVP(itsLogger), CEREAL_NVP(itsIdentifier),
		   CEREAL_NVP(itsUVRelativeToGrid), CEREAL_NVP(itsEarthShape));
	}
#endif
};

class regular_grid : public grid
{
   public:
	regular_grid();
	~regular_grid();
	regular_grid(const regular_grid&);
	regular_grid& operator=(const regular_grid& other) = delete;

	/* Return grid point value (incl. fractions) of a given latlon point */
	virtual point XY(const point& latlon) const = 0;

	/*
	 * Functions that are only valid for some grid types, but for ease
	 * of use they are declared here. It is up to the actual grid classes
	 * to implement correct functionality.
	 */

	virtual point BottomLeft() const = 0;
	virtual point TopRight() const = 0;

	virtual HPScanningMode ScanningMode() const;
	virtual void ScanningMode(HPScanningMode theScanningMode);

	virtual size_t Ni() const = 0;
	virtual size_t Nj() const = 0;

	virtual double Di() const = 0;
	virtual double Dj() const = 0;

   protected:
	bool EqualsTo(const regular_grid& other) const;

	HPScanningMode itsScanningMode;
#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(cereal::base_class<grid>(this), CEREAL_NVP(itsScanningMode));
	}
#endif
};

class irregular_grid : public grid
{
   public:
	irregular_grid();
	~irregular_grid();
	irregular_grid(const irregular_grid&);
	irregular_grid& operator=(const irregular_grid& other) = delete;

   protected:
	bool EqualsTo(const irregular_grid& other) const;

#ifdef SERIALIZATION
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

}  // namespace himan

#endif /* GRID_H */
