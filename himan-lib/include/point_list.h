/**
 * @file point_list.h
 *
 * @date Jan 23, 2013
 * @author partio
 */

#ifndef POINT_LIST_H
#define POINT_LIST_H

#include <string>
#include "point.h"
#include "grid.h"

namespace himan
{

class logger;
class point_list : public grid
{
	public:
		point_list();
		point_list(const std::vector<station>& theStations);

		virtual ~point_list() = default;

		point_list(const point_list& other);
		point_list& operator=(const point_list& other) = delete;

		std::string ClassName() const
		{
			return "himan::point_list";
		}

		std::ostream& Write(std::ostream& file) const;

		size_t Size() const override;
		size_t Ni() const override;
		size_t Nj() const override;
		double Di() const override;
		double Dj() const override;
		point FirstPoint() const override;
		point LastPoint() const override;
		
		bool Swap(HPScanningMode newScanningMode) override;
		
		point BottomLeft() const override;
		point TopRight() const override;
		
		bool operator==(const point_list& other) const;
		bool operator!=(const point_list& other) const;

		point LatLon(size_t locationIndex) const;
		
		const std::vector<station>& Stations() const;
		void Stations(const std::vector<station>& theStations);

		station Station(size_t locationIndex) const;
		void Station(size_t locationIndex, const station& theStation);
		
		HPScanningMode ScanningMode() const override;
		
		point_list* Clone() const override;
		
	private:
		bool EqualsTo(const point_list& other) const;

		std::vector<station> itsStations;
};


inline
std::ostream& operator<<(std::ostream& file, const point_list& ob)
{
	return ob.Write(file);
}

} // namespace himan

#endif /* POINT_LIST_H */
