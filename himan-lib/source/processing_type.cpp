#include "processing_type.h"

using namespace himan;

processing_type::processing_type(HPProcessingType theType) : itsType(theType)
{
}

processing_type::processing_type(HPProcessingType theType, double theValue, double theValue2)
    : itsType(theType), itsValue(theValue), itsValue2(theValue2)
{
}

processing_type::processing_type(HPProcessingType theType, double theValue, double theValue2,
                                 int theNumberOfEnsembleMembers)
    : itsType(theType), itsValue(theValue), itsValue2(theValue2), itsNumberOfEnsembleMembers(theNumberOfEnsembleMembers)
{
}

bool processing_type::operator==(const processing_type& other) const
{
	if (this == &other)
	{
		return true;
	}

	if (itsType != other.itsType || itsValue != other.itsValue || itsValue2 != other.itsValue2 ||
	    itsNumberOfEnsembleMembers != other.itsNumberOfEnsembleMembers)
	{
		return false;
	}

	return true;
}

bool processing_type::operator!=(const processing_type& other) const
{
	return !(*this == other);
}

HPProcessingType processing_type::Type() const
{
	return itsType;
}

void processing_type::Type(HPProcessingType theType)
{
	itsType = theType;
}

void processing_type::Value(double theValue)
{
	itsValue = theValue;
}

double processing_type::Value() const
{
	return itsValue;
}

void processing_type::Value2(double theValue2)
{
	itsValue2 = theValue2;
}

double processing_type::Value2() const
{
	return itsValue2;
}

int processing_type::NumberOfEnsembleMembers() const
{
	return itsNumberOfEnsembleMembers;
}

void processing_type::NumberOfEnsembleMembers(int theNumberOfEnsembleMembers)
{
	itsNumberOfEnsembleMembers = theNumberOfEnsembleMembers;
}

std::ostream& processing_type::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;

	file << "__itsType__ " << itsType << " (" << HPProcessingTypeToString.at(itsType) << ")" << std::endl;
	file << "__itsValue__ " << itsValue << std::endl;
	file << "__itsValue2__ " << itsValue2 << std::endl;
	file << "__itsNumberOfEnsembleMembers__ " << itsNumberOfEnsembleMembers << std::endl;

	return file;
}
