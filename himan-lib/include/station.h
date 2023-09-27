#pragma once

#include "point.h"
namespace himan
{
class station : public point
{
   public:
	station();
	station(int theId);
	station(int theId, double lon, double lat);
	station(int theId, const std::string& theName, double lon, double lat);

	bool operator==(const station& thePoint) const;
	bool operator!=(const station& thePoint) const;

	int Id() const;
	void Id(int theId);

	std::string Name() const;
	void Name(const std::string& theName);

	std::ostream& Write(std::ostream& file) const;

   private:
	int itsId;  // FMISID
	std::string itsName;
#ifdef HAVE_CEREAL
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(cereal::base_class<point>(this), CEREAL_NVP(itsId), CEREAL_NVP(itsName));
	}
#endif
};

inline std::ostream& operator<<(std::ostream& file, const station& ob)
{
	return ob.Write(file);
}
}  // namespace himan