#pragma once
#include "himan_common.h"
#include "serialization.h"
#include <fmt/format.h>
#include <ostream>

#ifndef EARTH_SHAPE_H
#define EARTH_SHAPE_H

namespace himan
{
template <typename T>
class earth_shape
{
   public:
	earth_shape();
	earth_shape(T r);  // sphere
	earth_shape(T theA, T theB);
	earth_shape(T theA, T theB, const std::string& theName);

	~earth_shape() = default;
	earth_shape(const earth_shape&) = default;

	bool operator==(const earth_shape& other) const;
	bool operator!=(const earth_shape& other) const;

	std::string ClassName() const
	{
		return "himan::earth_shape";
	}

	T A() const;
	void A(T theA);

	T B() const;
	void B(T theB);

	/**
	 * @brief Return flattening of the spheroid. Note that often inverse flattening is used.
	 */

	T F() const;

	/**
	 *  @brief Return squared eccentricity
	 */

	T E2() const;

	std::string Name() const;
	std::string Proj4String() const;

	std::ostream& Write(std::ostream& file) const;

   private:
	T itsA;               // size of major axis in meters
	T itsB;               // size of minor axis in meters
	std::string itsName;  // Name to describe "well-known" values

#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsA), CEREAL_NVP(itsB), CEREAL_NCP(itsName));
	}
#endif
};

template <typename T>
earth_shape<T>::earth_shape() : itsA(MissingValue<T>()), itsB(MissingValue<T>()), itsName()
{
}

template <typename T>
earth_shape<T>::earth_shape(T r) : itsA(r), itsB(r), itsName()
{
}

template <typename T>
earth_shape<T>::earth_shape(T theA, T theB) : itsA(theA), itsB(theB), itsName()
{
}

template <typename T>
earth_shape<T>::earth_shape(T theA, T theB, const std::string& theName) : itsA(theA), itsB(theB), itsName(theName)
{
}

template <typename T>
bool earth_shape<T>::operator==(const earth_shape<T>& other) const
{
	if (itsA == other.itsA && itsB == other.itsB)
	{
		return true;
	}

	// Check for missing values so that we can compare with default constructor
	return ((IsMissing(itsA) && IsMissing(other.itsA)) && (IsMissing(itsB) && IsMissing(other.itsB)));
}

template <typename T>
bool earth_shape<T>::operator!=(const earth_shape<T>& other) const
{
	return !(*this == other);
}

template <typename T>
T earth_shape<T>::A() const
{
	return itsA;
}

template <typename T>
void earth_shape<T>::A(T theA)
{
	itsA = theA;
}

template <typename T>
T earth_shape<T>::B() const
{
	return itsB;
}

template <typename T>
void earth_shape<T>::B(T theB)
{
	itsB = theB;
}

template <typename T>
T earth_shape<T>::F() const
{
	return (itsA - itsB) / itsA;
}

template <typename T>
T earth_shape<T>::E2() const
{
	return (itsA * itsA - itsB * itsB) / (itsA * itsA);
}

template <typename T>
std::string earth_shape<T>::Name() const
{
	return itsName;
}

template <typename T>
std::string earth_shape<T>::Proj4String() const
{
	if (itsName == "GRS80" || itsName == "WGS84")
	{
		return fmt::format("+ellps={}", itsName);
	}

	if (IsValid<T>(itsA) && IsValid<T>(itsB))
	{
		if (itsA == itsB)
		{
			return fmt::format("+R={}", itsA);
		}
		return fmt::format("+a={} +b={}", itsA, itsB);
	}

	return "";
}

template <typename T>
std::ostream& himan::earth_shape<T>::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;
	file << "__itsA__ " << std::fixed << itsA << std::endl;
	file << "__itsB__ " << std::fixed << itsB << std::endl;
	file << "__itsName__" << itsName << std::endl;
	return file;
}

template <typename T>
std::ostream& operator<<(std::ostream& file, const earth_shape<T>& ob)
{
	return ob.Write(file);
}

const earth_shape<double> ELLIPS_NEWBASE(6371220, 6371220, "newbase");
const earth_shape<double> ELLIPS_WGS84(6378137, 6356752.31424783, "WGS84");
const earth_shape<double> ELLIPS_GRS80(6378137, 6356752.31414028, "GRS80");

}  // namespace himan

#endif /* EARTH_SHAPE_H */
