#include "stereographic_grid.h"
#include <NFmiGrid.h>
#include <NFmiStereographicArea.h>

using namespace himan;
using namespace std;

stereographic_grid::stereographic_grid()
    : grid(kRegularGrid, kStereographic),
      itsBottomLeft(),
      itsTopRight(),
      itsOrientation(kHPMissingInt),
      itsDi(kHPMissingValue),
      itsDj(kHPMissingValue),
      itsNi(kHPMissingInt),
      itsNj(kHPMissingInt)
{
	itsLogger = logger("stereographic_grid");
}

stereographic_grid::stereographic_grid(HPScanningMode theScanningMode, point theBottomLeft, point theTopRight,
                                       double theOrientation)
    : grid(kRegularGrid, kStereographic, theScanningMode),
      itsBottomLeft(theBottomLeft),
      itsTopRight(theTopRight),
      itsOrientation(theOrientation)
{
	if (itsScanningMode != kBottomLeft)
	{
		throw runtime_error("Only bottom left is supported for stereographic grids");
	}

	itsLogger = logger("stereographic_grid");
}

stereographic_grid::stereographic_grid(const stereographic_grid& other)
    : grid(other),
      itsBottomLeft(other.itsBottomLeft),
      itsTopRight(other.itsTopRight),
      itsOrientation(other.itsOrientation),
      itsDi(other.itsDi),
      itsDj(other.itsDj),
      itsNi(other.itsNi),
      itsNj(other.itsNj)
{
	itsLogger = logger("stereographic_grid");
}

stereographic_grid::~stereographic_grid() = default;

size_t stereographic_grid::Size() const
{
	if (itsNi == static_cast<size_t>(kHPMissingInt) || itsNj == static_cast<size_t>(kHPMissingInt))
	{
		return kHPMissingInt;
	}

	return itsNi * itsNj;
}

void stereographic_grid::Ni(size_t theNi) { itsNi = theNi; }
void stereographic_grid::Nj(size_t theNj) { itsNj = theNj; }
size_t stereographic_grid::Ni() const { return itsNi; }
size_t stereographic_grid::Nj() const { return itsNj; }
void stereographic_grid::Di(double theDi) { itsDi = theDi; }
void stereographic_grid::Dj(double theDj) { itsDj = theDj; }
double stereographic_grid::Di() const { return itsDi; }
double stereographic_grid::Dj() const { return itsDj; }
void stereographic_grid::Orientation(double theOrientation) { itsOrientation = theOrientation; }
double stereographic_grid::Orientation() const { return itsOrientation; }
HPScanningMode stereographic_grid::ScanningMode() const { return itsScanningMode; }
void stereographic_grid::ScanningMode(HPScanningMode theScanningMode)
{
	if (theScanningMode != kBottomLeft)
	{
		throw runtime_error("Only bottom left is supported for stereographic grids");
	}

	itsScanningMode = theScanningMode;
}

void stereographic_grid::CreateAreaAndGrid() const
{
	NFmiPoint bl(itsBottomLeft.X(), itsBottomLeft.Y());
	auto area = unique_ptr<NFmiStereographicArea>(new NFmiStereographicArea(
	    bl, itsDi * (static_cast<double>(itsNi) - 1), itsDj * (static_cast<double>(itsNj) - 1), itsOrientation));

	itsStereGrid = unique_ptr<NFmiGrid>(new NFmiGrid(area.get(), itsNi, itsNj));
}

point stereographic_grid::XY(const point& latlon) const
{
	assert(itsScanningMode == kBottomLeft);

	if (!itsStereGrid)
	{
		CreateAreaAndGrid();
	}

	auto xy = itsStereGrid->LatLonToGrid(latlon.X(), latlon.Y());
	return point(xy.X(), xy.Y());
}

point stereographic_grid::LatLon(size_t locationIndex) const
{
	assert(itsScanningMode == kBottomLeft);

	if (!itsStereGrid)
	{
		CreateAreaAndGrid();
	}

	auto ll = itsStereGrid->LatLon(locationIndex);

	return point(ll.X(), ll.Y());
}

void stereographic_grid::BottomLeft(const point& theBottomLeft) { itsBottomLeft = theBottomLeft; }
void stereographic_grid::TopRight(const point& theTopRight) { itsTopRight = theTopRight; }
point stereographic_grid::BottomLeft() const { return itsBottomLeft; }
point stereographic_grid::TopRight() const { return itsTopRight; }
point stereographic_grid::FirstPoint() const
{
	switch (itsScanningMode)
	{
		case kBottomLeft:
			return itsBottomLeft;
			break;
		case kUnknownScanningMode:
			throw runtime_error("Scanning mode not set");
		default:
			throw runtime_error("Scanning mode not supported: " + HPScanningModeToString.at(itsScanningMode));
	}
}

point stereographic_grid::LastPoint() const
{
	switch (itsScanningMode)
	{
		case kBottomLeft:
			return itsTopRight;
			break;
		case kUnknownScanningMode:
			throw runtime_error("Scanning mode not set");
		default:
			throw runtime_error("Scanning mode not supported: " + HPScanningModeToString.at(itsScanningMode));
	}
}

bool stereographic_grid::Swap(HPScanningMode newScanningMode)
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

stereographic_grid* stereographic_grid::Clone() const { return new stereographic_grid(*this); }
ostream& stereographic_grid::Write(std::ostream& file) const
{
	grid::Write(file);

	file << itsBottomLeft;
	file << itsTopRight;
	file << "__itsNi__ " << itsNi << endl;
	file << "__itsNj__ " << itsNj << endl;
	file << "__itsDi__ " << itsDi << endl;
	file << "__itsDj__ " << itsDj << endl;

	file << "__itsOrientation__ " << itsOrientation << endl;

	return file;
}

bool stereographic_grid::operator!=(const grid& other) const { return !(other == *this); }
bool stereographic_grid::operator==(const grid& other) const
{
	const stereographic_grid* g = dynamic_cast<const stereographic_grid*>(&other);

	if (g)
	{
		return EqualsTo(*g);
	}

	return false;
}

bool stereographic_grid::EqualsTo(const stereographic_grid& other) const
{
	if (!grid::EqualsTo(other))
	{
		return false;
	}

	if (itsBottomLeft != other.BottomLeft())
	{
		itsLogger.Trace("BottomLeft does not match: X " + to_string(itsBottomLeft.X()) + " vs " +
		                to_string(other.BottomLeft().X()));
		itsLogger.Trace("BottomLeft does not match: Y " + to_string(itsBottomLeft.Y()) + " vs " +
		                to_string(other.BottomLeft().Y()));
		return false;
	}

	if (itsTopRight != other.TopRight())
	{
		itsLogger.Trace("TopRight does not match: X " + to_string(itsTopRight.X()) + " vs " +
		                to_string(other.TopRight().X()));
		itsLogger.Trace("TopRight does not match: Y " + to_string(itsTopRight.Y()) + " vs " +
		                to_string(other.TopRight().Y()));
		return false;
	}

	if (itsDi != other.Di())
	{
		itsLogger.Trace("Di does not match: " + to_string(itsDi) + " vs " + to_string(other.Di()));
		return false;
	}

	if (itsDj != other.Dj())
	{
		itsLogger.Trace("Dj does not match: " + to_string(itsDj) + " vs " + to_string(other.Dj()));
		return false;
	}

	if (itsNi != other.Ni())
	{
		itsLogger.Trace("Ni does not match: " + to_string(itsNi) + " vs " + to_string(other.Ni()));
		return false;
	}

	if (itsNj != other.Nj())
	{
		itsLogger.Trace("Nj does not match: " + to_string(itsNj) + " vs " + to_string(other.Nj()));
		return false;
	}

	if (itsOrientation != other.Orientation())
	{
		itsLogger.Trace("Orientation does not match: " + to_string(itsOrientation) + " vs " +
		                to_string(other.Orientation()));
		return false;
	}

	return true;
}
