/**
 * @file irregular_grid.h
 *
 * @date Jan 23, 2013
 * @author partio
 */

#ifndef IRREGULAR_GRID_H
#define IRREGULAR_GRID_H

#include <string>
#include "point.h"
#include "grid.h"

class NFmiGrid;

namespace himan
{

class logger;
class irregular_grid : public grid
{
	public:
		irregular_grid();
		irregular_grid(const std::vector<station>& theStations);

		~irregular_grid() = default;

		irregular_grid(const irregular_grid& other);
		irregular_grid& operator=(const irregular_grid& other) = delete;


		std::string ClassName() const
		{
			return "himan::irregular_grid";
		}

		std::ostream& Write(std::ostream& file) const;

		size_t Size() const;

		unpacked& Data();

		void Data(const unpacked& d);

		/**
		 * @return True if parameter UV components are irregular_grid relative, false if they are earth-relative.
		 * On parameters with no UV components this has no meaning.
		 */

		bool UVRelativeToGrid() const;
		void UVRelativeToGrid(bool theUVRelativeToGrid);

		/**
		 * @brief Set the data value pointed by the iterators with a new one
		 * @return True if assignment was succesfull
		 */

		virtual bool Value(size_t locationIndex, double theValue);

		/**
		 * @return Data value pointed by the iterators
		 */

		virtual double Value(size_t locationIndex) const;

		/**
		 * @return Projection type of this info
		 *
		 * One info can hold only one type of projection
		 */

		HPProjectionType Projection() const;
		void Projection(HPProjectionType theProjection);

		std::vector<double> AB() const;
		void AB(const std::vector<double>& theAB);

		point BottomLeft() const;
		point TopRight() const;
		
		void BottomLeft(const point& theBottomLeft) {};
		void TopRight(const point& theTopRight) {};

		void Orientation(double theOrientation);
		double Orientation() const;

		point SouthPole() const;
		void SouthPole(const point& theSouthPole);

		bool operator==(const grid& other) const;
		bool operator!=(const grid& other) const;

		point LatLon(size_t locationIndex) const;

		bool IsPackedData() const
		{
			return false;
		}
		
		const std::vector<station>& Stations() const;
		void Stations(const std::vector<station>& theStations);

		station Station(size_t locationIndex) const;
		void Station(size_t locationIndex, const station& theStation);
		
	private:
		bool EqualsTo(const irregular_grid& other) const;

		unpacked itsData; //<! Variable to hold unpacked data

		HPScanningMode itsScanningMode; //<! When data is read from files, we need to know what is the scanning mode

		bool itsUVRelativeToGrid; //<! If true, wind UV components are relative to irregular_grid north and east (ie. are not earth-relative)

		HPProjectionType itsProjection;
		std::vector<double> itsAB;

		point itsBottomLeft;
		point itsTopRight;
		point itsSouthPole;

		double itsOrientation;

		std::unique_ptr<logger> itsLogger;

		std::vector<station> itsStations;
};

inline
std::ostream& operator<<(std::ostream& file, const irregular_grid& ob)
{
	return ob.Write(file);
}

} // namespace himan

#endif /* IRREGULAR_GRID_H */
