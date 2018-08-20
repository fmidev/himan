/**
 * @file grid.cpp
 *
 */

#include "grid.h"
#include "simple_packed.h"

using namespace himan;
using namespace std;

grid::grid()
    : itsData(0, 0, 1, MissingDouble()),
      itsGridClass(kUnknownGridClass),
      itsGridType(kUnknownGridType),
      itsIdentifier(),
      itsAB(),
      itsPackedData(),
      itsUVRelativeToGrid(false),
      itsEarthShape()
{
}

grid::~grid()
{
}
grid::grid(const grid& other)
    : itsData(other.itsData),
      itsGridClass(other.itsGridClass),
      itsGridType(other.itsGridType),
      itsIdentifier(other.itsIdentifier),
      itsAB(other.itsAB),
      itsPackedData(),
      itsUVRelativeToGrid(other.itsUVRelativeToGrid),
      itsEarthShape(other.itsEarthShape)
{
#ifdef HAVE_CUDA
	if (other.itsPackedData)
	{
		switch (other.itsPackedData->packingType)
		{
			case kSimplePacking:
				itsPackedData = unique_ptr<simple_packed>(
				    new simple_packed(*dynamic_cast<simple_packed*>(other.itsPackedData.get())));
				break;

			default:
				itsPackedData = unique_ptr<packed_data>(new packed_data(*itsPackedData));
				break;
		}
	}
#endif
}

bool grid::EqualsTo(const grid& other) const
{
	if (other.itsGridType != itsGridType)
	{
		itsLogger.Trace("Grid type does not match: " + HPGridTypeToString.at(itsGridType) + " vs " +
		                HPGridTypeToString.at(other.Type()));
		return false;
	}

	// Comparison of earth shape turned off for now
	if (false && other.itsEarthShape != itsEarthShape)
	{
		itsLogger.Trace("Earth shape does not match: A: " + to_string(other.itsEarthShape.A()) + " vs " +
		                to_string(itsEarthShape.A()) + " and B: " + to_string(other.itsEarthShape.B()) + " vs " +
		                to_string(itsEarthShape.B()));
		return false;
	}

	if (other.itsGridClass != itsGridClass)
	{
		return false;
	}

	if (other.itsIdentifier != itsIdentifier)
	{
		return false;
	}

	// We DON'T test for AB !
	// Why?

	return true;
}

packed_data& grid::PackedData()
{
	ASSERT(itsPackedData);
	return *itsPackedData;
}

void grid::PackedData(unique_ptr<packed_data> thePackedData)
{
	itsPackedData = move(thePackedData);
}
HPGridType grid::Type() const
{
	return itsGridType;
}
void grid::Type(HPGridType theGridType)
{
	itsGridType = theGridType;
}
HPGridClass grid::Class() const
{
	return itsGridClass;
}
void grid::Class(HPGridClass theGridClass)
{
	itsGridClass = theGridClass;
}
std::string grid::Identifier() const
{
	return itsIdentifier;
}
void grid::Identifier(const std::string& theIdentifier)
{
	itsIdentifier = theIdentifier;
}
bool grid::IsPackedData() const
{
	if (itsPackedData && itsPackedData->HasData())
	{
		return true;
	}

	return false;
}

matrix<double>& grid::Data()
{
	return itsData;
}
void grid::Data(const matrix<double>& theData)
{
	itsData = theData;
}
ostream& grid::Write(std::ostream& file) const
{
	file << "<" << ClassName() << ">" << std::endl;

	file << "__itsUVRelativeToGrid__ " << itsUVRelativeToGrid << std::endl;
	file << "__itsAB__";

	for (size_t i = 0; i < itsAB.size(); i++)
	{
		file << " " << itsAB[i];
	}

	file << std::endl;

	file << itsEarthShape;
	file << "__IsPackedData__ " << IsPackedData() << std::endl;
	file << itsData;

	return file;
}

size_t grid::Size() const
{
	throw runtime_error("grid::Size() called");
}
void grid::Value(size_t theLocationIndex, double theValue)
{
	itsData.Set(theLocationIndex, theValue);
}
double grid::Value(size_t theLocationIndex) const
{
	return double(itsData.At(theLocationIndex));
}
vector<double> grid::AB() const
{
	return itsAB;
}
void grid::AB(const vector<double>& theAB)
{
	itsAB = theAB;
}
point grid::LatLon(size_t theLocationIndex) const
{
	throw runtime_error("grid::LatLon() called");
}
bool grid::operator!=(const grid& other) const
{
	return !(other == *this);
}
bool grid::operator==(const grid& other) const
{
	return EqualsTo(other);
}
bool grid::UVRelativeToGrid() const
{
	return itsUVRelativeToGrid;
}
void grid::UVRelativeToGrid(bool theUVRelativeToGrid)
{
	itsUVRelativeToGrid = theUVRelativeToGrid;
}

earth_shape<double> grid::EarthShape() const
{
	return itsEarthShape;
}

void grid::EarthShape(const earth_shape<double>& theEarthShape)
{
	itsEarthShape = theEarthShape;
}

//--------------- regular grid
regular_grid::regular_grid() : grid(), itsScanningMode(kUnknownScanningMode)
{
	Class(kRegularGrid);
}

regular_grid::~regular_grid()
{
}
regular_grid::regular_grid(const regular_grid& other) : grid(other), itsScanningMode(other.itsScanningMode)
{
}

bool regular_grid::EqualsTo(const regular_grid& other) const
{
	if (!grid::EqualsTo(other))
	{
		return false;
	}

	if (other.itsScanningMode != itsScanningMode)
	{
		return false;
	}

	return true;
}

HPScanningMode regular_grid::ScanningMode() const
{
	return itsScanningMode;
}
void regular_grid::ScanningMode(HPScanningMode theScanningMode)
{
	itsScanningMode = theScanningMode;
}

//--------------- irregular grid

irregular_grid::irregular_grid() : grid()
{
	Class(kIrregularGrid);
}

irregular_grid::~irregular_grid()
{
}

irregular_grid::irregular_grid(const irregular_grid& other) : grid(other)
{
}

bool irregular_grid::EqualsTo(const irregular_grid& other) const
{
	if (!grid::EqualsTo(other))
	{
		return false;
	}

	return true;
}
