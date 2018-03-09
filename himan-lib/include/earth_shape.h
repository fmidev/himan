#pragma once
#include "serialization.h"
#include <ostream>

namespace himan
{
class earth_shape
{
   public:
	earth_shape();
	earth_shape(double theA, double theB);
	~earth_shape() = default;
	earth_shape(const earth_shape& other) = default;

	bool operator==(const earth_shape& other) const;
	bool operator!=(const earth_shape& other) const;

	std::string ClassName() const
	{
		return "himan::earth_shape";
	}

	double A() const;
	void A(double theA);

	double B() const;
	void B(double theB);

	/**
	 * @brief Return flattening of the spheroid. Note that often inverse flattening is used.
	 */

	double F() const;

	std::ostream& Write(std::ostream& file) const;

   private:
	double itsA;  // size of major axis in meters
	double itsB;  // size of minor axis in meters

#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsA), CEREAL_NVP(itsB))
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const earth_shape& ob)
{
	return ob.Write(file);
}

}  // namespace himan
