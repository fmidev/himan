#include <fmt/core.h>
#include <forecast_type.h>

using namespace himan;

forecast_type::forecast_type(HPForecastType theType) : itsForecastType(theType), itsForecastTypeValue(kHPMissingValue)
{
}

forecast_type::forecast_type(HPForecastType theType, double theValue)
    : itsForecastType(theType), itsForecastTypeValue(theValue)
{
}

HPForecastType forecast_type::Type() const
{
	return itsForecastType;
}
void forecast_type::Type(HPForecastType theForecastType)
{
	itsForecastType = theForecastType;
}
double forecast_type::Value() const
{
	return itsForecastTypeValue;
}
void forecast_type::Value(double theValue)
{
	itsForecastTypeValue = theValue;
}
std::ostream& forecast_type::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;
	file << "__itsForecastType__ " << HPForecastTypeToString.at(itsForecastType) << std::endl;
	file << "__itsForecastTypeValue__ " << itsForecastTypeValue << std::endl;

	return file;
}

bool forecast_type::operator==(const forecast_type& other) const
{
	if (this == &other)
	{
		return true;
	}

	return (itsForecastType == other.itsForecastType && itsForecastTypeValue == other.itsForecastTypeValue);
}

bool forecast_type::operator!=(const forecast_type& other) const
{
	return !(*this == other);
}
forecast_type::operator std::string() const
{
	return fmt::format("{}/{}", HPForecastTypeToString.at(itsForecastType), itsForecastTypeValue);
}
