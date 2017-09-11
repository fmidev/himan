#include "latitude_longitude_grid.h"
#include "info.h"
#include <NFmiRotatedLatLonArea.h>

#ifdef HAVE_CUDA
#include "simple_packed.h"
#endif

using namespace himan;
using namespace std;

latitude_longitude_grid::latitude_longitude_grid()
    : grid(kRegularGrid, kLatitudeLongitude),
      itsBottomLeft(),
      itsTopRight(),
      itsBottomRight(),
      itsTopLeft(),
      itsDi(kHPMissingValue),
      itsDj(kHPMissingValue),
      itsNi(kHPMissingInt),
      itsNj(kHPMissingInt),
      itsIsGlobal(false)
{
	itsLogger = logger("latitude_longitude_grid");
}

latitude_longitude_grid::latitude_longitude_grid(HPScanningMode theScanningMode, point theBottomLeft, point theTopRight)
    : grid(kRegularGrid, kLatitudeLongitude, theScanningMode),
      itsBottomLeft(theBottomLeft),
      itsTopRight(theTopRight),
      itsBottomRight(),
      itsTopLeft(),
      itsDi(kHPMissingValue),
      itsDj(kHPMissingValue),
      itsNi(kHPMissingInt),
      itsNj(kHPMissingInt),
      itsIsGlobal(false)
{
	itsLogger = logger("latitude_longitude_grid");
	UpdateCoordinates();
}

latitude_longitude_grid::latitude_longitude_grid(const latitude_longitude_grid& other)
    : grid(other),
      itsBottomLeft(other.itsBottomLeft),
      itsTopRight(other.itsTopRight),
      itsBottomRight(other.itsBottomRight),
      itsTopLeft(other.itsTopLeft),
      itsDi(other.itsDi),
      itsDj(other.itsDj),
      itsNi(other.itsNi),
      itsNj(other.itsNj),
      itsIsGlobal(other.itsIsGlobal)
{
	itsLogger = logger("latitude_longitude_grid");
}

size_t latitude_longitude_grid::Size() const
{
	if (itsNi == static_cast<size_t>(kHPMissingInt) || itsNj == static_cast<size_t>(kHPMissingInt))
	{
		return kHPMissingInt;
	}

	return itsNi * itsNj;
}

point latitude_longitude_grid::TopRight() const { return itsTopRight; }
point latitude_longitude_grid::BottomLeft() const { return itsBottomLeft; }
point latitude_longitude_grid::TopLeft() const { return itsTopLeft; }
point latitude_longitude_grid::BottomRight() const { return itsBottomRight; }
void latitude_longitude_grid::TopRight(const point& theTopRight)
{
	itsTopRight = theTopRight;
	UpdateCoordinates();
}

void latitude_longitude_grid::BottomLeft(const point& theBottomLeft)
{
	itsBottomLeft = theBottomLeft;
	UpdateCoordinates();
}

void latitude_longitude_grid::TopLeft(const point& theTopLeft)
{
	itsTopLeft = theTopLeft;
	UpdateCoordinates();
}

void latitude_longitude_grid::BottomRight(const point& theBottomRight)
{
	itsBottomRight = theBottomRight;
	UpdateCoordinates();
}

point latitude_longitude_grid::FirstPoint() const
{
	switch (itsScanningMode)
	{
		case kTopLeft:
			return itsTopLeft;
		case kBottomLeft:
			return itsBottomLeft;
		default:
			return point();
	}
}

point latitude_longitude_grid::LastPoint() const
{
	switch (itsScanningMode)
	{
		case kTopLeft:
			return itsBottomRight;
		case kBottomLeft:
			return itsTopRight;
		default:
			return point();
	}
}

point latitude_longitude_grid::XY(const himan::point& latlon) const
{
	double x = (latlon.X() - itsBottomLeft.X()) / Di();

	if (itsIsGlobal && (x < 0 || x > static_cast<double>(Ni()) - 1))
	{
		// wrap x if necessary
		// this might happen f.ex. with EC where grid start at 0 meridian and
		// we interpolate from say -10 to 40 longitude

		while (x < 0)
		{
			x += static_cast<double>(Ni());
		}
		while (x > static_cast<double>(Ni()) - 1)
		{
			x -= static_cast<double>(Ni()) - 1.;
		}
	}

	double y;

	switch (itsScanningMode)
	{
		case kBottomLeft:
			y = (latlon.Y() - itsBottomLeft.Y()) / Dj();
			break;
		case kTopLeft:
			y = (itsTopLeft.Y() - latlon.Y()) / Dj();
			break;
		default:
			throw runtime_error("Scanning mode not supported: " + HPScanningModeToString.at(itsScanningMode));
			break;
	}

	return point(x, y);
}

point latitude_longitude_grid::LatLon(size_t locationIndex) const
{
	assert(itsNi != static_cast<size_t>(kHPMissingInt));
	assert(itsNj != static_cast<size_t>(kHPMissingInt));
	assert(!IsKHPMissingValue(Di()));
	assert(!IsKHPMissingValue(Dj()));
	assert(locationIndex < itsNi * itsNj);

	point firstPoint = FirstPoint();

	double j = floor(static_cast<double>(locationIndex / itsNi));
	double i = fmod(static_cast<double>(locationIndex), static_cast<double>(itsNi));

	point ret(firstPoint.X() + i * Di(), kHPMissingInt);

	switch (itsScanningMode)
	{
		case kBottomLeft:
			ret.Y(firstPoint.Y() + j * Dj());
			break;

		case kTopLeft:
			ret.Y(firstPoint.Y() - j * Dj());
			break;
		default:
			throw runtime_error("Scanning mode not supported: " + HPScanningModeToString.at(itsScanningMode));
	}

	return ret;
}

bool latitude_longitude_grid::Swap(HPScanningMode newScanningMode)
{
	if (itsScanningMode == newScanningMode)
	{
		return true;
	}

	// Flip with regards to x axis

	if ((itsScanningMode == kTopLeft && newScanningMode == kBottomLeft) ||
	    (itsScanningMode == kBottomLeft && newScanningMode == kTopLeft))
	{
		size_t halfSize = static_cast<size_t>(floor(Nj() / 2));

		for (size_t y = 0; y < halfSize; y++)
		{
			for (size_t x = 0; x < Ni(); x++)
			{
				double upper = itsData.At(x, y);
				double lower = itsData.At(x, Nj() - 1 - y);

				itsData.Set(x, y, 0, lower);
				itsData.Set(x, Nj() - 1 - y, 0, upper);
			}
		}
	}
	else
	{
		itsLogger.Error("Swap from mode " + string(HPScanningModeToString.at(itsScanningMode)) + " to mode " +
		                string(HPScanningModeToString.at(newScanningMode)) + " not implemented yet");
		return false;
	}

	itsScanningMode = newScanningMode;

	return true;
}

void latitude_longitude_grid::Ni(size_t theNi) { itsNi = theNi; }
void latitude_longitude_grid::Nj(size_t theNj) { itsNj = theNj; }
size_t latitude_longitude_grid::Ni() const { return itsNi; }
size_t latitude_longitude_grid::Nj() const { return itsNj; }
void latitude_longitude_grid::Di(double theDi) { itsDi = theDi; }
void latitude_longitude_grid::Dj(double theDj) { itsDj = theDj; }
double latitude_longitude_grid::Di() const
{
	if (IsKHPMissingValue(itsDi) && itsNi != static_cast<size_t>(kHPMissingInt) &&
	    !IsKHPMissingValue(FirstPoint().X()) && !IsKHPMissingValue(LastPoint().X()))
	{
		double fx = FirstPoint().X();
		double lx = LastPoint().X();

		if (fx == 0. && lx < 0) lx += 360;
		itsDi = fabs((fx - lx) / (static_cast<double>(itsNi) - 1.));
	}

	return itsDi;
}

double latitude_longitude_grid::Dj() const
{
	if (IsKHPMissingValue(itsDj) && itsNj != static_cast<size_t>(kHPMissingInt) &&
	    !IsKHPMissingValue(FirstPoint().X()) && !IsKHPMissingValue(LastPoint().X()))
	{
		itsDj = fabs((FirstPoint().Y() - LastPoint().Y()) / (static_cast<double>(itsNj) - 1.));
	}

	return itsDj;
}

void latitude_longitude_grid::FirstPoint(const point& theFirstPoint)
{
	switch (itsScanningMode)
	{
		case kBottomLeft:
			itsBottomLeft = theFirstPoint;
			break;
		case kTopLeft:
			itsTopLeft = theFirstPoint;
			break;
		default:
			break;
	}
	UpdateCoordinates();
}

void latitude_longitude_grid::LastPoint(const point& theLastPoint)
{
	switch (itsScanningMode)
	{
		case kBottomLeft:
			itsTopRight = theLastPoint;
			break;
		case kTopLeft:
			itsBottomRight = theLastPoint;
			break;
		default:
			break;
	}
	UpdateCoordinates();
}

void latitude_longitude_grid::UpdateCoordinates() const
{
	const point missing;

	if ((itsBottomLeft != missing && itsTopRight != missing) && (itsTopLeft == missing || itsBottomRight == missing))
	{
		if (itsBottomLeft.X() == 0. && itsTopRight.X() < 0)
		{
			itsTopRight.X(itsTopRight.X() + 360.);
		}

		itsTopLeft = point(itsBottomLeft.X(), itsTopRight.Y());
		itsBottomRight = point(itsTopRight.X(), itsBottomLeft.Y());
	}
	else if ((itsBottomLeft == missing || itsTopRight == missing) &&
	         (itsTopLeft != missing && itsBottomRight != missing))
	{
		if (itsTopLeft.X() == 0. && itsBottomRight.X() < 0)
		{
			itsBottomRight.X(itsBottomRight.X() + 360.);
		}

		itsBottomLeft = point(itsTopLeft.X(), itsBottomRight.Y());
		itsTopRight = point(itsBottomRight.X(), itsTopLeft.Y());
	}

	Di();
	Dj();

	if (FirstPoint() != missing && LastPoint() != missing && !IsKHPMissingValue(Di()))
	{
		double span = itsBottomLeft.X() + itsTopRight.X() + Di();
		itsIsGlobal = (span == 0. || span == 360.);
	}
}

bool latitude_longitude_grid::operator!=(const grid& other) const { return !(other == *this); }
bool latitude_longitude_grid::operator==(const grid& other) const
{
	const latitude_longitude_grid* g = dynamic_cast<const latitude_longitude_grid*>(&other);

	if (g)
	{
		return EqualsTo(*g);
	}

	return false;
}

bool latitude_longitude_grid::EqualsTo(const latitude_longitude_grid& other) const
{
	if (!grid::EqualsTo(other))
	{
		return false;
	}

	if (itsBottomLeft != other.BottomLeft())
	{
		itsLogger.Trace("BottomLeft does not match: X " + boost::lexical_cast<std::string>(itsBottomLeft.X()) + " vs " +
		                boost::lexical_cast<std::string>(other.BottomLeft().X()));
		itsLogger.Trace("BottomLeft does not match: Y " + boost::lexical_cast<std::string>(itsBottomLeft.Y()) + " vs " +
		                boost::lexical_cast<std::string>(other.BottomLeft().Y()));
		return false;
	}

	if (itsTopRight != other.TopRight())
	{
		itsLogger.Trace("TopRight does not match: X " + boost::lexical_cast<std::string>(itsTopRight.X()) + " vs " +
		                boost::lexical_cast<std::string>(other.TopRight().X()));
		itsLogger.Trace("TopRight does not match: Y " + boost::lexical_cast<std::string>(itsTopRight.Y()) + " vs " +
		                boost::lexical_cast<std::string>(other.TopRight().Y()));
		return false;
	}

	const double kEpsilon = 0.0001;

	if (fabs(Di() - other.Di()) > kEpsilon)
	{
		itsLogger.Trace("Di does not match: " + boost::lexical_cast<std::string>(Di()) + " vs " +
		                boost::lexical_cast<std::string>(other.Di()));
		return false;
	}

	if (fabs(Dj() - other.Dj()) > kEpsilon)
	{
		itsLogger.Trace("Dj does not match: " + boost::lexical_cast<std::string>(Dj()) + " vs " +
		                boost::lexical_cast<std::string>(other.Dj()));
		return false;
	}

	if (itsNi != other.Ni())
	{
		itsLogger.Trace("Ni does not match: " + boost::lexical_cast<std::string>(itsNi) + " vs " +
		                boost::lexical_cast<std::string>(other.Ni()));
		return false;
	}

	if (itsNj != other.Nj())
	{
		itsLogger.Trace("Nj does not match: " + boost::lexical_cast<std::string>(itsNj) + " vs " +
		                boost::lexical_cast<std::string>(other.Nj()));
		return false;
	}

	return true;
}

latitude_longitude_grid* latitude_longitude_grid::Clone() const { return new latitude_longitude_grid(*this); }
ostream& latitude_longitude_grid::Write(std::ostream& file) const
{
	grid::Write(file);

	file << itsBottomLeft << itsTopLeft << itsTopRight << itsBottomRight << "__itsNi__ " << itsNi << endl
	     << "__itsNj__ " << itsNj << endl
	     << "__itsDi__ " << itsDi << endl
	     << "__itsDj__ " << itsDj << endl
	     << "__itsIsGlobal__" << itsIsGlobal << endl;

	return file;
}

rotated_latitude_longitude_grid::rotated_latitude_longitude_grid() : latitude_longitude_grid(), itsSouthPole()
{
	itsGridType = kRotatedLatitudeLongitude;
	itsLogger = logger("rotated_latitude_longitude_grid");
}

rotated_latitude_longitude_grid::rotated_latitude_longitude_grid(HPScanningMode theScanningMode, point theBottomLeft,
                                                                 point theTopRight, point theSouthPole,
                                                                 bool initiallyRotated)
    : latitude_longitude_grid(theScanningMode, theBottomLeft, theTopRight), itsSouthPole(theSouthPole)
{
	if (!initiallyRotated)
		throw std::runtime_error("Unable to create rotated_latitude_longitude_grid with unrotated coordinates, yet");

	itsGridType = kRotatedLatitudeLongitude;
	itsLogger = logger("rotated_latitude_longitude_grid");
}

rotated_latitude_longitude_grid::rotated_latitude_longitude_grid(const rotated_latitude_longitude_grid& other)
    : latitude_longitude_grid(other), itsSouthPole(other.itsSouthPole)
{
	itsLogger = logger("rotated_latitude_longitude_grid");
}

rotated_latitude_longitude_grid::~rotated_latitude_longitude_grid() = default;

bool rotated_latitude_longitude_grid::operator!=(const grid& other) const { return !(other == *this); }
bool rotated_latitude_longitude_grid::operator==(const grid& other) const
{
	const rotated_latitude_longitude_grid* g = dynamic_cast<const rotated_latitude_longitude_grid*>(&other);

	if (g)
	{
		return EqualsTo(*g);
	}

	return false;
}

bool rotated_latitude_longitude_grid::EqualsTo(const rotated_latitude_longitude_grid& other) const
{
	if (!grid::EqualsTo(other) || !latitude_longitude_grid::EqualsTo(other))
	{
		return false;
	}

	if (itsSouthPole != other.SouthPole())
	{
		itsLogger.Trace("SouthPole does not match: X " + boost::lexical_cast<std::string>(itsSouthPole.X()) + " vs " +
		                boost::lexical_cast<std::string>(other.SouthPole().X()));
		itsLogger.Trace("SouthPole does not match: Y " + boost::lexical_cast<std::string>(itsSouthPole.Y()) + " vs " +
		                boost::lexical_cast<std::string>(other.SouthPole().Y()));
		return false;
	}

	// Note! We DON'T test for uv relative to grid!
	// if (itsUVRelativeToGrid != other.UVRelativeToGrid())
	//{
	//	itsLogger.Trace("UVRelativeToGrid does not match: " + boost::lexical_cast<std::string> (itsUVRelativeToGrid) +
	//" vs " + boost::lexical_cast<std::string> (other.UVRelativeToGrid()));
	//	return false;
	//}

	return true;
}

rotated_latitude_longitude_grid* rotated_latitude_longitude_grid::Clone() const
{
	return new rotated_latitude_longitude_grid(*this);
}

point rotated_latitude_longitude_grid::SouthPole() const { return itsSouthPole; }
void rotated_latitude_longitude_grid::SouthPole(const point& theSouthPole) { itsSouthPole = theSouthPole; }
point rotated_latitude_longitude_grid::XY(const point& latlon) const
{
	if (!itsRotLatLonArea)
	{
		assert(itsBottomLeft != point());
		assert(itsTopRight != point());
		assert(itsSouthPole != point());

		itsRotLatLonArea = unique_ptr<NFmiRotatedLatLonArea>(new NFmiRotatedLatLonArea(
		    NFmiPoint(itsBottomLeft.X(), itsBottomLeft.Y()), NFmiPoint(itsTopRight.X(), itsTopRight.Y()),
		    NFmiPoint(itsSouthPole.X(), itsSouthPole.Y()), NFmiPoint(0., 0.), NFmiPoint(1., 1.), true));
	}

	auto rotpoint = itsRotLatLonArea->ToRotLatLon(NFmiPoint(latlon.X(), latlon.Y()));

	return latitude_longitude_grid::XY(point(rotpoint.X(), rotpoint.Y()));
}

point rotated_latitude_longitude_grid::LatLon(size_t locationIndex) const
{
	point rll = latitude_longitude_grid::LatLon(locationIndex);  // rotated coordinates

	if (!itsRotLatLonArea)
	{
		assert(itsBottomLeft != point());
		assert(itsTopRight != point());

		itsRotLatLonArea = unique_ptr<NFmiRotatedLatLonArea>(new NFmiRotatedLatLonArea(
		    NFmiPoint(itsBottomLeft.X(), itsBottomLeft.Y()), NFmiPoint(itsTopRight.X(), itsTopRight.Y()),
		    NFmiPoint(itsSouthPole.X(), itsSouthPole.Y()), NFmiPoint(0., 0.), NFmiPoint(1., 1.), true));
	}

	auto regpoint = itsRotLatLonArea->ToRegLatLon(NFmiPoint(rll.X(), rll.Y()));

	return point(regpoint.X(), regpoint.Y());
}

point rotated_latitude_longitude_grid::RotatedLatLon(size_t locationIndex) const
{
	return latitude_longitude_grid::LatLon(locationIndex);
}

ostream& rotated_latitude_longitude_grid::Write(std::ostream& file) const
{
	latitude_longitude_grid::Write(file);

	file << itsSouthPole;

	return file;
}
