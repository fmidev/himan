/*
 * param.cpp
 *
 */

#include "param.h"

using namespace himan;
using namespace std;

param::~param() {}
param::param()
    : itsId(kHPMissingInt),
      itsName("XX-X"),
      itsScale(1),
      itsBase(0),
      itsUnivId(kHPMissingInt),
      itsGribParameter(kHPMissingInt),
      itsGribCategory(kHPMissingInt),
      itsGribDiscipline(kHPMissingInt),
      itsGribTableVersion(kHPMissingInt),
      itsGribIndicatorOfParameter(kHPMissingInt),
      itsVersion(1),
      itsInterpolationMethod(kBiLinear),
      itsUnit(kUnknownUnit),
      itsAggregation(),
      itsPrecision(kHPMissingInt)
{
}

param::param(const string& theName, unsigned long theUnivId)
    : itsId(kHPMissingInt),
      itsName(theName),
      itsScale(1),
      itsBase(0),
      itsUnivId(theUnivId),
      itsGribParameter(kHPMissingInt),
      itsGribCategory(kHPMissingInt),
      itsGribDiscipline(kHPMissingInt),
      itsGribTableVersion(kHPMissingInt),
      itsGribIndicatorOfParameter(kHPMissingInt),
      itsVersion(1),
      itsInterpolationMethod(kBiLinear),
      itsUnit(kUnknownUnit),
      itsAggregation(),
      itsPrecision(kHPMissingInt)
{
}

param::param(const string& theName, unsigned long theUnivId, HPParameterUnit theUnit)
    : itsId(kHPMissingInt),
      itsName(theName),
      itsScale(1),
      itsBase(0),
      itsUnivId(theUnivId),
      itsGribParameter(kHPMissingInt),
      itsGribCategory(kHPMissingInt),
      itsGribDiscipline(kHPMissingInt),
      itsGribTableVersion(kHPMissingInt),
      itsGribIndicatorOfParameter(kHPMissingInt),
      itsVersion(1),
      itsInterpolationMethod(kBiLinear),
      itsUnit(theUnit),
      itsAggregation(),
      itsPrecision(kHPMissingInt)
{
}

param::param(const string& theName)
    : itsId(kHPMissingInt),
      itsName(theName),
      itsScale(1),
      itsBase(0),
      itsUnivId(kHPMissingInt),
      itsGribParameter(kHPMissingInt),
      itsGribCategory(kHPMissingInt),
      itsGribDiscipline(kHPMissingInt),
      itsGribTableVersion(kHPMissingInt),
      itsGribIndicatorOfParameter(kHPMissingInt),
      itsVersion(1),
      itsInterpolationMethod(kBiLinear),
      itsUnit(kUnknownUnit),
      itsAggregation(),
      itsPrecision(kHPMissingInt)
{
}

param::param(const string& theName, unsigned long theUnivId, double theScale, double theBase,
             HPInterpolationMethod theInterpolationMethod)
    : itsId(kHPMissingInt),
      itsName(theName),
      itsScale(theScale),
      itsBase(theBase),
      itsUnivId(theUnivId),
      itsGribParameter(kHPMissingInt),
      itsGribCategory(kHPMissingInt),
      itsGribDiscipline(kHPMissingInt),
      itsGribTableVersion(kHPMissingInt),
      itsGribIndicatorOfParameter(kHPMissingInt),
      itsVersion(1),
      itsInterpolationMethod(theInterpolationMethod),
      itsUnit(kUnknownUnit),
      itsAggregation(),
      itsPrecision(kHPMissingInt)
{
}

param::param(const string& theName, unsigned long theUnivId, long theGribDiscipline, long theGribCategory,
             long theGribParameter)
    : itsId(kHPMissingInt),
      itsName(theName),
      itsScale(1),
      itsBase(0),
      itsUnivId(theUnivId),
      itsGribParameter(theGribParameter),
      itsGribCategory(theGribCategory),
      itsGribDiscipline(theGribDiscipline),
      itsGribTableVersion(kHPMissingInt),
      itsGribIndicatorOfParameter(kHPMissingInt),
      itsVersion(1),
      itsInterpolationMethod(kBiLinear),
      itsUnit(kUnknownUnit),
      itsAggregation(),
      itsPrecision(kHPMissingInt)
{
}

param::param(const param& other)
    : itsId(other.itsId),
      itsName(other.itsName),
      itsScale(other.itsScale),
      itsBase(other.itsBase),
      itsUnivId(other.itsUnivId),
      itsGribParameter(other.itsGribParameter),
      itsGribCategory(other.itsGribCategory),
      itsGribDiscipline(other.itsGribDiscipline),
      itsGribTableVersion(other.itsGribTableVersion),
      itsGribIndicatorOfParameter(other.itsGribIndicatorOfParameter),
      itsVersion(other.itsVersion),
      itsInterpolationMethod(other.itsInterpolationMethod),
      itsUnit(other.itsUnit),
      itsAggregation(other.itsAggregation),
      itsPrecision(other.itsPrecision)
{
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
	itsAggregation = other.itsAggregation;
	itsPrecision = other.itsPrecision;

	return *this;
}

bool param::operator==(const param& other) const
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

	if (UnivId() != static_cast<unsigned int>(kHPMissingInt) &&
	    other.UnivId() != static_cast<unsigned int>(kHPMissingInt) && UnivId() != other.UnivId())
	{
		return false;
	}

	// Grib 1

	if (itsGribTableVersion != kHPMissingInt && other.GribTableVersion() != kHPMissingInt &&
	    itsGribTableVersion != other.GribTableVersion())
	{
		return false;
	}

	if (itsGribIndicatorOfParameter != kHPMissingInt && other.GribIndicatorOfParameter() != kHPMissingInt &&
	    itsGribIndicatorOfParameter != other.GribIndicatorOfParameter())
	{
		return false;
	}

	// Grib 2

	if (itsGribDiscipline != kHPMissingInt && other.GribDiscipline() != kHPMissingInt &&
	    itsGribDiscipline != other.GribDiscipline())
	{
		return false;
	}

	if (itsGribCategory != kHPMissingInt && other.GribCategory() != kHPMissingInt &&
	    itsGribCategory != other.GribCategory())
	{
		return false;
	}

	if (itsGribParameter != kHPMissingInt && other.GribParameter() != kHPMissingInt &&
	    itsGribParameter != other.GribParameter())
	{
		return false;
	}

	if (itsAggregation.Type() != kUnknownAggregationType && other.itsAggregation.Type() != kUnknownAggregationType &&
	    itsAggregation != other.itsAggregation)
	{
		return false;
	}

	if (itsVersion != other.itsVersion)
	{
		return false;
	}

	return true;
}

bool param::operator!=(const param& other) const { return !(*this == other); }
void param::GribParameter(long theGribParameter) { itsGribParameter = theGribParameter; }
long param::GribParameter() const { return itsGribParameter; }
void param::GribDiscipline(long theGribDiscipline) { itsGribDiscipline = theGribDiscipline; }
long param::GribDiscipline() const { return itsGribDiscipline; }
void param::GribCategory(long theGribCategory) { itsGribCategory = theGribCategory; }
long param::GribCategory() const { return itsGribCategory; }
void param::GribIndicatorOfParameter(long theGribIndicatorOfParameter)
{
	itsGribIndicatorOfParameter = theGribIndicatorOfParameter;
}

long param::GribIndicatorOfParameter() const { return itsGribIndicatorOfParameter; }
unsigned long param::UnivId() const { return itsUnivId; }
void param::UnivId(unsigned long theUnivId) { itsUnivId = theUnivId; }
string param::Name() const { return itsName; }
void param::Name(const string& theName) { itsName = theName; }
HPParameterUnit param::Unit() const { return itsUnit; }
void param::Unit(HPParameterUnit theUnit) { itsUnit = theUnit; }
void param::GribTableVersion(long theVersion) { itsGribTableVersion = theVersion; }
long param::GribTableVersion() const { return itsGribTableVersion; }
const aggregation& param::Aggregation() const { return itsAggregation; }
void param::Aggregation(const aggregation& theAggregation) { itsAggregation = theAggregation; }
double param::Base() const { return itsBase; }
void param::Base(double theBase) { itsBase = theBase; }
double param::Scale() const { return itsScale; }
void param::Scale(double theScale) { itsScale = theScale; }
long param::Id() const { return itsId; }
void param::Id(long theId) { itsId = theId; }
HPInterpolationMethod param::InterpolationMethod() const { return itsInterpolationMethod; }
void param::InterpolationMethod(HPInterpolationMethod theInterpolationMethod)
{
	itsInterpolationMethod = theInterpolationMethod;
}
int param::Precision() const { return itsPrecision; }
void param::Precision(int thePrecision) { itsPrecision = thePrecision; }
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
	file << "__itsGribTableVersion__ " << itsGribTableVersion << endl;
	file << "__itsIndicatorOfParameter__ " << itsGribIndicatorOfParameter << endl;
	file << "__itsUnit__ " << static_cast<int>(itsUnit) << endl;
	file << "__itsVersion__ " << itsVersion << endl;
	file << "__itsInterpolationMethod__ " << HPInterpolationMethodToString.at(itsInterpolationMethod) << endl;
	file << "__itsPrecision__" << itsPrecision << endl;
	file << itsAggregation;

	return file;
}
