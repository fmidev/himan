/*
 * param.cpp
 *
 *  Created on: Nov 27, 2012
 *      Author: partio
 */

#include "param.h"
#include "logger_factory.h"

using namespace himan;
using namespace std;

param::param()
	: itsParam(shared_ptr<NFmiParam> (new NFmiParam(kHPMissingInt, "HimanDefaultParam")))
	, itsGribParameter(kHPMissingInt)
	, itsGribCategory(kHPMissingInt)
	, itsGribDiscipline(kHPMissingInt)
	, itsGribTableVersion(kHPMissingInt)
	, itsGribIndicatorOfParameter(kHPMissingInt)
	, itsUnit(kUnknownUnit)
{
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("param"));
}

param::param(const std::string& theName, unsigned long theUnivId)
	: itsParam(shared_ptr<NFmiParam> (new NFmiParam(theUnivId, NFmiString(theName))))
	, itsGribParameter(kHPMissingInt)
	, itsGribCategory(kHPMissingInt)
	, itsGribDiscipline(kHPMissingInt)
	, itsGribTableVersion(kHPMissingInt)
	, itsGribIndicatorOfParameter(kHPMissingInt)
	, itsUnit(kUnknownUnit)
{
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("param"));
}

param::param(const std::string& theName)
	: itsParam(shared_ptr<NFmiParam> (new NFmiParam(kHPMissingInt, NFmiString(theName))))
	, itsGribParameter(kHPMissingInt)
	, itsGribCategory(kHPMissingInt)
	, itsGribDiscipline(kHPMissingInt)
	, itsGribTableVersion(kHPMissingInt)
	, itsGribIndicatorOfParameter(kHPMissingInt)
	, itsUnit(kUnknownUnit)
{
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("param"));
}

param::param(const std::string& theName,
			 unsigned long theUnivId,
			 float theScale,
			 float theBase,
			 const std::string& thePrecision,
			 FmiInterpolationMethod theInterpolationMethod)
	: itsParam(shared_ptr<NFmiParam> (new NFmiParam (theUnivId,
									  NFmiString(theName),
									  kFloatMissing,
									  kFloatMissing,
									  theScale,
									  theBase,
									  NFmiString(thePrecision),
									  theInterpolationMethod)))
	, itsGribParameter(kHPMissingInt)
	, itsGribCategory(kHPMissingInt)
	, itsGribDiscipline(kHPMissingInt)
	, itsGribTableVersion(kHPMissingInt)
	, itsGribIndicatorOfParameter(kHPMissingInt)
	, itsUnit(kUnknownUnit)
{
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("param"));
}

param::param(const param& other)
	: itsParam(shared_ptr<NFmiParam> (new NFmiParam (*other.itsParam)))
	, itsGribParameter(other.itsGribParameter)
	, itsGribCategory(other.itsGribCategory)
	, itsGribDiscipline(other.itsGribDiscipline)
	, itsGribTableVersion(other.itsGribTableVersion)
	, itsGribIndicatorOfParameter(other.itsGribIndicatorOfParameter)
	, itsUnit(other.itsUnit)
{
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("param"));
}

param& param::operator=(const param& other)
{
	itsParam = shared_ptr<NFmiParam> (new NFmiParam (*other.itsParam));
	itsGribParameter = other.itsGribParameter;
	itsGribCategory = other.itsGribCategory;
	itsGribDiscipline = other.itsGribDiscipline;
	itsGribTableVersion = other.itsGribTableVersion;
	itsGribIndicatorOfParameter = other.itsGribIndicatorOfParameter;
	itsUnit = other.itsUnit;

	return *this;
}

bool param::operator==(const param& other)
{
	if (this == &other)
	{
		return true;
	}

	if (Name() != other.Name())
	{
		return false;
	}

	if (UnivId() != static_cast<unsigned int> (kHPMissingInt) && other.UnivId() !=  static_cast<unsigned int> (kHPMissingInt) && UnivId() != other.UnivId())
	{
		return false;
	}

	// Grib 1

	if (itsGribTableVersion != kHPMissingInt && other.GribTableVersion() != kHPMissingInt && itsGribTableVersion != other.GribTableVersion())
	{
		return false;
	}

	if (itsGribIndicatorOfParameter != kHPMissingInt && other.GribIndicatorOfParameter() != kHPMissingInt && itsGribIndicatorOfParameter != other.GribIndicatorOfParameter())
	{
		return false;
	}

	// Grib 2

	if (itsGribDiscipline != kHPMissingInt && other.GribDiscipline() != kHPMissingInt && itsGribDiscipline != other.GribDiscipline())
	{
		return false;
	}

	if (itsGribCategory != kHPMissingInt && other.GribCategory() != kHPMissingInt && itsGribCategory != other.GribCategory())
	{
		return false;
	}

	if (itsGribParameter != kHPMissingInt && other.GribParameter() != kHPMissingInt && itsGribParameter != other.GribParameter())
	{
		return false;
	}

	return true;
}

bool param::operator!=(const param& other)
{
	return !(*this == other);
}

void param::GribParameter(long theGribParameter)
{
	itsGribParameter = theGribParameter;
}

long param::GribParameter() const
{
	return itsGribParameter;
}

void param::GribDiscipline(long theGribDiscipline)
{
	itsGribDiscipline = theGribDiscipline;
}

long param::GribDiscipline() const
{
	return itsGribDiscipline;
}

void param::GribCategory(long theGribCategory)
{
	itsGribCategory = theGribCategory;
}

long param::GribCategory() const
{
	return itsGribCategory;
}

void param::GribIndicatorOfParameter(long theGribIndicatorOfParameter)
{
	itsGribIndicatorOfParameter = theGribIndicatorOfParameter;
}

long param::GribIndicatorOfParameter() const
{
	return itsGribIndicatorOfParameter;
}

unsigned long param::UnivId() const
{
	return itsParam->GetIdent();
}

void param::UnivId(unsigned long theUnivId)
{
	itsParam->SetIdent(theUnivId);
}

std::string param::Name() const
{
	return std::string(itsParam->GetName());
}

void param::Name(std::string theName)
{
	itsParam->SetName(NFmiString(theName));
}

HPParameterUnit param::Unit() const
{
	return itsUnit;
}

void param::Unit(HPParameterUnit theUnit)
{
	itsUnit = theUnit;
}

void param::GribTableVersion(long theVersion)
{
	itsGribTableVersion = theVersion;
}

long param::GribTableVersion() const
{
	return itsGribTableVersion;
}

std::ostream& param::Write(std::ostream& file) const
{

	file << "<" << ClassName() << " " << Version() << ">" << std::endl;
	file << "__itsName__ " << std::string(itsParam->GetName()) << std::endl;
	file << "__itsUnivId__ " << itsParam->GetIdent() << std::endl;
	file << "__itsGribParameter__ " << itsGribParameter << std::endl;
	file << "__itsGribCategory__ " << itsGribCategory << std::endl;
	file << "__itsGribDiscipline__ " << itsGribDiscipline << std::endl;
	file << "__itsScale__ " << itsParam->Scale() << std::endl;
	file << "__itsBase__ " << itsParam->Base() << std::endl;
	file << "__itsUnit__ " << itsUnit << std::endl;

	return file;
}
