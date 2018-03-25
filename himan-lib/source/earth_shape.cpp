#include "earth_shape.h"
#include "himan_common.h"

using namespace himan;

earth_shape::earth_shape() : itsA(MissingDouble()), itsB(MissingDouble())
{
}

earth_shape::earth_shape(double r) : itsA(r), itsB(r)
{
}

earth_shape::earth_shape(double theA, double theB) : itsA(theA), itsB(theB)
{
}

bool earth_shape::operator==(const earth_shape& other) const
{
	if (itsA == other.itsA && itsB == other.itsB)
	{
		return true;
	}

	// Check for missing values so that we can compare with default constructor
	return ((IsMissing(itsA) && IsMissing(other.itsA)) && (IsMissing(itsB) && IsMissing(other.itsB)));
}

bool earth_shape::operator!=(const earth_shape& other) const
{
	return !(*this == other);
}

double earth_shape::A() const
{
	return itsA;
}

void earth_shape::A(double theA)
{
	itsA = theA;
}

double earth_shape::B() const
{
	return itsB;
}

void earth_shape::B(double theB)
{
	itsB = theB;
}

double earth_shape::F() const
{
	return (itsA - itsB) / itsA;
}

std::ostream& himan::earth_shape::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;
	file << "__itsA__ " << std::fixed << itsA << std::endl;
	file << "__itsB__ " << std::fixed << itsB << std::endl;

	return file;
}
