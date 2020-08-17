#include "station.h"

using namespace himan;

std::ostream& station::Write(std::ostream& file) const
{
	point::Write(file);
	file << "__itsId__ " << itsId << std::endl;
	file << "__itsName__ " << itsName << std::endl;

	return file;
}

station::station() : point(), itsId(kHPMissingInt), itsName("Himan default station")
{
}
station::station(int theId) : point(), itsId(theId), itsName("Himan default station")
{
}
station::station(int theId, const std::string& theName, double lon, double lat)
    : point(lon, lat), itsId(theId), itsName(theName)
{
}

station::station(int theId, double lon, double lat) : point(lon, lat), itsId(theId)
{
}
bool station::operator==(const station& other) const
{
	return (point::LatLonCompare(*this, other) && itsName == other.itsName && itsId == other.itsId);
}

bool station::operator!=(const station& other) const
{
	return !(*this == other);
}
int station::Id() const
{
	return itsId;
}
void station::Id(int theId)
{
	itsId = theId;
}
std::string station::Name() const
{
	return itsName;
}
void station::Name(const std::string& theName)
{
	itsName = theName;
}
