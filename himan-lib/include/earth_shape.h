#pragma once
#include "himan_common.h"
#include "serialization.h"
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

	std::ostream& Write(std::ostream& file) const;

   private:
	T itsA;  // size of major axis in meters
	T itsB;  // size of minor axis in meters

#ifdef SERIALIZATION
	friend class cereal::access;

	template <class Archive>
	void serialize(Archive& ar)
	{
		ar(CEREAL_NVP(itsA), CEREAL_NVP(itsB));
	}
#endif
};

template <typename T>
earth_shape<T>::earth_shape() : itsA(MissingValue<T>()), itsB(MissingValue<T>())
{
}

template <typename T>
earth_shape<T>::earth_shape(T r) : itsA(r), itsB(r)
{
}

template <typename T>
earth_shape<T>::earth_shape(T theA, T theB) : itsA(theA), itsB(theB)
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
std::ostream& himan::earth_shape<T>::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;
	file << "__itsA__ " << std::fixed << itsA << std::endl;
	file << "__itsB__ " << std::fixed << itsB << std::endl;

	return file;
}

template <typename T>
std::ostream& operator<<(std::ostream& file, const earth_shape<T>& ob)
{
	return ob.Write(file);
}

}  // namespace himan

#endif /* EARTH_SHAPE_H */
