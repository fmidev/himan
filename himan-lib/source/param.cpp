/*
 * param.cpp
 *
 */

#include "param.h"

using namespace himan;
using namespace std;

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
      itsPrecision(kHPMissingInt),
      itsProcessingType()
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
      itsPrecision(kHPMissingInt),
      itsProcessingType()
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
      itsPrecision(kHPMissingInt),
      itsProcessingType()
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
      itsPrecision(kHPMissingInt),
      itsProcessingType()
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
      itsPrecision(kHPMissingInt),
      itsProcessingType()
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
      itsPrecision(kHPMissingInt),
      itsProcessingType()
{
}

param::param(const map<string, string>& databaseInfo) : param()
{
	if (!databaseInfo.empty())
	{
		itsId = stoi(databaseInfo.at("id"));
		itsName = databaseInfo.at("name");
		itsVersion = stoi(databaseInfo.at("version"));

		try
		{
			itsGribIndicatorOfParameter = stoi(databaseInfo.at("grib1_number"));
			itsGribTableVersion = stoi(databaseInfo.at("grib1_table_version"));
		}
		catch (const out_of_range& e)
		{
		}
		catch (const invalid_argument& e)
		{
		}

		try
		{
			itsGribDiscipline = stoi(databaseInfo.at("grib2_discipline"));
			itsGribCategory = stoi(databaseInfo.at("grib2_category"));
			itsGribParameter = stoi(databaseInfo.at("grib2_number"));
		}
		catch (const out_of_range& e)
		{
		}
		catch (const invalid_argument& e)
		{
		}

		try
		{
			itsPrecision = stoi(databaseInfo.at("precision"));
		}
		catch (const out_of_range& e)
		{
		}
		catch (const invalid_argument& e)
		{
		}

		try
		{
			itsUnivId = stoi(databaseInfo.at("univ_id"));
			itsScale = stod(databaseInfo.at("scale"));
			itsBase = stod(databaseInfo.at("base"));
		}
		catch (const out_of_range& e)
		{
		}
		catch (const invalid_argument& e)
		{
		}
	}
}

bool param::operator==(const param& other) const
{
	if (this == &other)
	{
		return true;
	}

	if (itsName != other.itsName)
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

	if (itsProcessingType != other.itsProcessingType)
	{
		return false;
	}

	return true;
}

bool param::operator!=(const param& other) const
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
const aggregation& param::Aggregation() const
{
	return itsAggregation;
}
void param::Aggregation(const aggregation& theAggregation)
{
	itsAggregation = theAggregation;
}
processing_type& param::ProcessingType()
{
	return itsProcessingType;
}
const processing_type& param::ProcessingType() const
{
	return itsProcessingType;
}
void param::ProcessingType(const processing_type& theProcessingType)
{
	itsProcessingType = theProcessingType;
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
int param::Precision() const
{
	return itsPrecision;
}
void param::Precision(int thePrecision)
{
	itsPrecision = thePrecision;
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
	file << "__itsGribTableVersion__ " << itsGribTableVersion << endl;
	file << "__itsIndicatorOfParameter__ " << itsGribIndicatorOfParameter << endl;
	file << "__itsUnit__ " << static_cast<int>(itsUnit) << endl;
	file << "__itsVersion__ " << itsVersion << endl;
	file << "__itsInterpolationMethod__ " << HPInterpolationMethodToString.at(itsInterpolationMethod) << endl;
	file << "__itsPrecision__" << itsPrecision << endl;
	file << itsAggregation;
	file << itsProcessingType;

	return file;
}
