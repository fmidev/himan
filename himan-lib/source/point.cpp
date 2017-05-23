#include "point.h"

std::ostream& himan::point::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;
	file << "__itsX__ " << itsX << std::endl;
	file << "__itsY__ " << itsY << std::endl;

	return file;
}