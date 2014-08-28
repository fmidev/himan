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

param::~param() {}

param::param()
	: itsId(kHPMissingInt)
	, itsName("XX-X")
	, itsScale(1)
	, itsBase(0)
	, itsUnivId(kHPMissingInt)
	, itsGribParameter(kHPMissingInt)
	, itsGribCategory(kHPMissingInt)
	, itsGribDiscipline(kHPMissingInt)
	, itsGribTableVersion(kHPMissingInt)
	, itsGribIndicatorOfParameter(kHPMissingInt)
	, itsVersion(1)
	, itsInterpolationMethod(kUnknownInterpolationMethod)
	, itsUnit(kUnknownUnit)
	, itsMissingValue(kHPMissingValue)
	, itsAggregation()
{
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("param"));
}

param::param(const string& theName, unsigned long theUnivId)
	: itsId(kHPMissingInt)
	, itsName(theName)
	, itsScale(1)
	, itsBase(0)
	, itsUnivId(theUnivId)
	, itsGribParameter(kHPMissingInt)
	, itsGribCategory(kHPMissingInt)
	, itsGribDiscipline(kHPMissingInt)
	, itsGribTableVersion(kHPMissingInt)
	, itsGribIndicatorOfParameter(kHPMissingInt)
	, itsVersion(1)
	, itsInterpolationMethod(kUnknownInterpolationMethod)
	, itsUnit(kUnknownUnit)
	, itsMissingValue(kHPMissingValue)
	, itsAggregation()
{
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("param"));
}

param::param(const string& theName)
	: itsId(kHPMissingInt)
	, itsName(theName)
	, itsScale(1)
	, itsBase(0)
	, itsUnivId(kHPMissingInt)
	, itsGribParameter(kHPMissingInt)
	, itsGribCategory(kHPMissingInt)
	, itsGribDiscipline(kHPMissingInt)
	, itsGribTableVersion(kHPMissingInt)
	, itsGribIndicatorOfParameter(kHPMissingInt)
	, itsVersion(1)
	, itsInterpolationMethod(kUnknownInterpolationMethod)
	, itsUnit(kUnknownUnit)
	, itsMissingValue(kHPMissingValue)
	, itsAggregation()
{
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("param"));
}

param::param(const string& theName,
			 unsigned long theUnivId,
			 double theScale,
			 double theBase,
			 HPInterpolationMethod theInterpolationMethod)
	: itsId(kHPMissingInt)
	, itsName(theName)
	, itsScale(theScale)
	, itsBase(theBase)
	, itsUnivId(theUnivId)
	, itsGribParameter(kHPMissingInt)
	, itsGribCategory(kHPMissingInt)
	, itsGribDiscipline(kHPMissingInt)
	, itsGribTableVersion(kHPMissingInt)
	, itsGribIndicatorOfParameter(kHPMissingInt)
	, itsVersion(1)
	, itsInterpolationMethod(theInterpolationMethod)
	, itsUnit(kUnknownUnit)
	, itsMissingValue(kHPMissingValue)
	, itsAggregation()
{
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("param"));
}

param::param(const string& theName, unsigned long theUnivId, long theGribDiscipline, long theGribCategory, long theGribParameter)
	: itsId(kHPMissingInt)
	, itsName(theName)
	, itsScale(1)
	, itsBase(0)
	, itsUnivId(theUnivId)
	, itsGribParameter(theGribParameter)
	, itsGribCategory(theGribCategory)
	, itsGribDiscipline(theGribDiscipline)
	, itsGribTableVersion(kHPMissingInt)
	, itsGribIndicatorOfParameter(kHPMissingInt)
	, itsVersion(1)
	, itsInterpolationMethod(kUnknownInterpolationMethod)
	, itsUnit(kUnknownUnit)
	, itsMissingValue(kHPMissingValue)
	, itsAggregation()
{
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("param"));
}

param::param(const param& other)
	: itsId(other.itsId)
	, itsName(other.itsName)
	, itsScale(other.itsScale)
	, itsBase(other.itsBase)
	, itsUnivId(other.itsUnivId)
	, itsGribParameter(other.itsGribParameter)
	, itsGribCategory(other.itsGribCategory)
	, itsGribDiscipline(other.itsGribDiscipline)
	, itsGribTableVersion(other.itsGribTableVersion)
	, itsGribIndicatorOfParameter(other.itsGribIndicatorOfParameter)
	, itsVersion(other.itsVersion)
	, itsInterpolationMethod(other.itsInterpolationMethod)
	, itsUnit(other.itsUnit)
	, itsMissingValue(other.itsMissingValue)
	, itsAggregation(other.itsAggregation)
{
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("param"));
}

param& param::operator=(const param& other)
{
	itsId = other.itsId;
	itsName = other.itsName;
	itsScale = other.itsScale;
	itsBase = other.itsBase;
	itsUnivId = other.itsUnivId;
	itsGribParameter = other.itsGribParameter;
	itsGribCategory = other.itsGribCategory;
	itsGribDiscipline = other.itsGribDiscipline;
	itsGribTableVersion = other.itsGribTableVersion;
	itsGribIndicatorOfParameter = other.itsGribIndicatorOfParameter;
	itsVersion = other.itsVersion;
	itsInterpolationMethod = other.itsInterpolationMethod;
	itsUnit = other.itsUnit;
	itsMissingValue = other.itsMissingValue;
	itsAggregation = other.itsAggregation;

	return *this;
}

bool param::operator==(const param& other)
{
	if (this == &other)
	{
		return true;
	}

	if (itsId != other.itsId)
	{
		return false;
	}
	
	if (itsName != other.itsName)
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

	if (itsAggregation.Type() != kUnknownAggregationType && other.itsAggregation.Type() != kUnknownAggregationType && itsAggregation != other.itsAggregation)
	{
		return false;
	}

	if (itsVersion != other.itsVersion)
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
	return itsUnivId;
}

void param::UnivId(unsigned long theUnivId)
{
	itsUnivId = theUnivId;
}

string param::Name() const
{
	return itsName;
}

void param::Name(const string& theName)
{
	itsName = theName;
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

aggregation& param::Aggregation()
{
	return itsAggregation;
}

double param::Base() const
{
	return itsBase;
}

void param::Base(double theBase)
{
	itsBase = theBase;
}

double param::Scale() const
{
	return itsScale;
}

void param::Scale(double theScale)
{
	itsScale = theScale;
}

long param::Id() const
{
	return itsId;
}

void param::Id(long theId)
{
	itsId = theId;
}

HPInterpolationMethod param::InterpolationMethod() const
{
	return itsInterpolationMethod;
}

void param::InterpolationMethod(HPInterpolationMethod theInterpolationMethod)
{
	itsInterpolationMethod = theInterpolationMethod;
}


ostream& param::Write(ostream& file) const
{

	file << "<" << ClassName() << ">" << endl;
	file << "__itsName__ " << itsName << endl;
	file << "__itsScale__ " << itsScale << endl;
	file << "__itsBase__ " << itsBase << endl;
	file << "__itsUnivId__ " << itsUnivId << endl;
	file << "__itsGribParameter__ " << itsGribParameter << endl;
	file << "__itsGribCategory__ " << itsGribCategory << endl;
	file << "__itsGribDiscipline__ " << itsGribDiscipline << endl;
	file << "__itsUnit__ " << itsUnit << endl;
	file << "__itsVersion__ " << itsVersion << endl;
	file << "__itsInterpolationMethod__ " << HPInterpolationMethodToString.at(itsInterpolationMethod) << endl;

	file << itsAggregation;

	return file;
}
