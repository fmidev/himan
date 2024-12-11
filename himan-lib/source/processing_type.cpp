#include "processing_type.h"
#include "util.h"

using namespace himan;

processing_type::processing_type(HPProcessingType theType) : itsType(theType)
{
}

processing_type::processing_type(HPProcessingType theType, double theValue) : itsType(theType), itsValue(theValue)
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

processing_type::processing_type(const std::string& procstr) : processing_type()
{
	if (procstr.empty())
	{
		return;
	}

	const auto tokens = util::Split(procstr, ",");

	itsType = HPStringToProcessingType.at(tokens[0]);

	if (tokens.size() > 1)
	{
		itsValue = std::stod(tokens[1]);
	}
	if (tokens.size() == 3)
	{
		itsValue2 = std::stod(tokens[2]);
	}
}

bool processing_type::operator==(const processing_type& other) const
{
	if (this == &other)
	{
		return true;
	}

	// When comparing type, allow kProbabilityGreaterThan and kProbabilityGreaterThanOrEqual to match
	// because grib2, our primary file type, does not distinguish these two.
	// Same goes for kProbabilityLessThan and kProbabilityLessThanOrEqual.
	const bool typeMatch = (itsType == other.itsType ||
	                        (itsType == kProbabilityGreaterThan && other.itsType == kProbabilityGreaterThanOrEqual) ||
	                        (itsType == kProbabilityGreaterThanOrEqual && other.itsType == kProbabilityGreaterThan) ||
	                        (itsType == kProbabilityLessThan && other.itsType == kProbabilityLessThanOrEqual) ||
	                        (itsType == kProbabilityLessThanOrEqual && other.itsType == kProbabilityLessThan));

	const bool value1Match = (!itsValue && !other.itsValue) ||
	                         ((itsValue && other.itsValue) && fabs(itsValue.value() - other.itsValue.value()) < 0.0001);
	const bool value2Match =
	    (!itsValue2 && !other.itsValue2) ||
	    ((itsValue2 && other.itsValue2) && fabs(itsValue2.value() - other.itsValue2.value()) < 0.0001);

	// Disregard number of ensemble members in comparison, because that's not a
	// defining element for a processing type. It's more like extra meta data.
	if (typeMatch == false || value1Match == false || value2Match == false)
	//	    itsNumberOfEnsembleMembers != other.itsNumberOfEnsembleMembers)
	{
		return false;
	}

	return true;
}

bool processing_type::operator!=(const processing_type& other) const
{
	return !(*this == other);
}

processing_type::operator std::string() const
{
	return fmt::format("{}/{}/{}/{}", HPProcessingTypeToString.at(itsType), itsValue.value_or(MissingDouble()),
	                   itsValue2.value_or(MissingDouble()), itsNumberOfEnsembleMembers);
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

std::optional<double> processing_type::Value() const
{
	return itsValue;
}

void processing_type::Value2(double theValue2)
{
	itsValue2 = theValue2;
}

std::optional<double> processing_type::Value2() const
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

	file << "__itsType__ " << static_cast<int>(itsType) << " (" << HPProcessingTypeToString.at(itsType) << ")"
	     << std::endl;
	file << "__itsValue__ " << itsValue.value_or(MissingDouble()) << std::endl;
	file << "__itsValue2__ " << itsValue2.value_or(MissingDouble()) << std::endl;
	file << "__itsNumberOfEnsembleMembers__ " << itsNumberOfEnsembleMembers << std::endl;

	return file;
}
