/**
 * @file grid.h
 *
 * @date Jan 23, 2013
 * @author partio
 */

#ifndef GRID_H
#define GRID_H

/**
 * @class grid
 *
 * @brief Parent class for grids
 */

#include <string>
#include "himan_common.h"
#include "point.h"

#include "matrix.h"

namespace himan
{

class grid
{
	public:
		grid();
		virtual ~grid();

		/**
		 * @brief Copy constructor for grid
		 *
		 * When grid is copied, the contents (ie. class d_matrix_t) is copied as
		 * well.
		 *
		 * @param other 
		 */
		
		//grid(const grid& other);
		grid& operator=(const grid& other) = delete;


		virtual std::string ClassName() const
		{
			return "himan::grid";
		}

		virtual HPGridType Type() const
		{
			return itsGridType;
		}
		
		virtual std::ostream& Write(std::ostream& file) const = 0;

		/**
		 * 
		 * @return  Grid size
		 */
		
		virtual size_t Size() const = 0;

		/**
		 *
		 * @return Data matrix
		 */

		virtual matrix<double>& Data() = 0;

		/**
		 * @brief Replace current data matrix with the function argument
		 * @param d shared pointer to a data matrix
		 */

		virtual void Data(const matrix<double>& d) = 0;


		//HPScanningMode ScanningMode() const;
		//void ScanningMode(HPScanningMode theScanningMode);

		/**
		 * @return True if parameter UV components are grid relative, false if they are earth-relative.
		 * On parameters with no UV components this has no meaning.
		 */

		virtual bool UVRelativeToGrid() const = 0;
		virtual void UVRelativeToGrid(bool theUVRelativeToGrid) = 0;

		/**
		 * @brief Set the data value pointed by the iterators with a new one
		 * @return True if assignment was succesfull
		 */

		virtual bool Value(size_t locationIndex, double theValue) = 0;

		/**
		 * @return Data value pointed by the iterators
		 */

		virtual double Value(size_t locationIndex) const = 0;

		/**
		 * @return Projection type of this info
		 *
		 * One info can hold only one type of projection
		 */

		virtual HPProjectionType Projection() const = 0;
		virtual void Projection(HPProjectionType theProjection) = 0;

		virtual std::vector<double> AB() const = 0;
		virtual void AB(const std::vector<double>& theAB) = 0;

		virtual point BottomLeft() const = 0;
		virtual point TopRight() const = 0;

		virtual void BottomLeft(const point& theBottomLeft) = 0;
		virtual void TopRight(const point& theTopRight) = 0;

		virtual void Orientation(double theOrientation) = 0;
		virtual double Orientation() const = 0;

		virtual point SouthPole() const = 0;
		virtual void SouthPole(const point& theSouthPole) = 0;

		/**
		 * @brief Create a newbase grid (NFmiGrid) from current data
		 * @return Raw pointer to NFmiGrid
		 */

		//NFmiGrid* ToNewbaseGrid() const;

		virtual bool IsPackedData() const = 0;

		virtual point LatLon(size_t locationIndex) const = 0;
		virtual bool operator==(const grid& other) const = 0;
		virtual bool operator!=(const grid& other) const = 0;
		
	protected:
	
		bool EqualsTo(const grid& other) const;

		HPGridType itsGridType;

};

inline
std::ostream& operator<<(std::ostream& file, const grid& ob)
{
	return ob.Write(file);
}

} // namespace himan

#endif /* GRID_H */
