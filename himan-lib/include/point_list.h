/**
 * @file point_list.h
 *
 */

#ifndef POINT_LIST_H
#define POINT_LIST_H

#include "grid.h"
#include "serialization.h"
#include "station.h"
#include <string>

namespace himan
{
class logger;

class point_list : public irregular_grid
{
   public:
	point_list();
	explicit point_list(const std::vector<station>& theStations);

	virtual ~point_list(){};

	point_list(const point_list& other);
	point_list& operator=(const point_list& other) = delete;

	std::string ClassName() const override
	{
		return "himan::point_list";
	}
	std::ostream& Write(std::ostream& file) const;

	size_t Size() const override;
	point FirstPoint() const override;
	point LastPoint() const override;

	bool operator==(const grid& other) const;
	bool operator!=(const grid& other) const;

	point LatLon(size_t locationIndex) const;

	const std::vector<station>& Stations() const;
	void Stations(const std::vector<station>& theStations);

	station Station(size_t locationIndex) const;
	void Station(size_t locationIndex, const station& theStation);

	size_t Hash() const override;

	std::unique_ptr<grid> Clone() const override;

   private:
	bool EqualsTo(const point_list& other) const;

	std::vector<station> itsStations;
#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(cereal::base_class<irregular_grid>(this), CEREAL_NVP(itsStations));
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const point_list& ob)
{
	return ob.Write(file);
}
}  // namespace himan

#ifdef SERIALIZATION
CEREAL_REGISTER_TYPE(himan::point_list);
#endif

#endif /* POINT_LIST_H */
