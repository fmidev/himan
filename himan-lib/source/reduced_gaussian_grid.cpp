/**
 * @file reduced_gaussian_grid.cpp
 *
 */

#include "reduced_gaussian_grid.h"
#include "logger.h"
#include "numerical_functions.h"

#include <algorithm>
#include <numeric>

using namespace himan;
using namespace std;

reduced_gaussian_grid::reduced_gaussian_grid()
    : irregular_grid(), itsNumberOfPointsAlongParallels(), itsAccumulatedPointsAlongParallels(), itsLatitudes(), itsN(0)
{
	itsLogger = logger("reduced_gaussian_grid");
	Type(kReducedGaussian);
}

reduced_gaussian_grid::reduced_gaussian_grid(const reduced_gaussian_grid& other)
    : irregular_grid(other),
      itsNumberOfPointsAlongParallels(other.itsNumberOfPointsAlongParallels),
      itsAccumulatedPointsAlongParallels(other.itsAccumulatedPointsAlongParallels),
      itsLatitudes(other.itsLatitudes),
      itsN(other.itsN)
{
	itsLogger = logger("reduced_gaussian_grid");
}

int reduced_gaussian_grid::N() const
{
	return itsN;
}

void reduced_gaussian_grid::N(int theN)
{
	itsN = theN;
	itsLatitudes = GetLatitudes(theN);
}

size_t reduced_gaussian_grid::Size() const
{
	return itsAccumulatedPointsAlongParallels.back();
}

std::vector<int> reduced_gaussian_grid::NumberOfPointsAlongParallels() const
{
	return itsNumberOfPointsAlongParallels;
}

void reduced_gaussian_grid::NumberOfPointsAlongParallels(const std::vector<int>& theNumberOfPointsAlongParallels)
{
	ASSERT((itsN == kHPMissingInt && itsNumberOfPointsAlongParallels.size() == 0) ||
	       static_cast<size_t>(itsN * 2) == theNumberOfPointsAlongParallels.size());
	itsNumberOfPointsAlongParallels = theNumberOfPointsAlongParallels;

	// recompute the accumulated points
	itsAccumulatedPointsAlongParallels.resize(theNumberOfPointsAlongParallels.size() + 1);
	std::partial_sum(theNumberOfPointsAlongParallels.begin(), theNumberOfPointsAlongParallels.end(),
	                 ++itsAccumulatedPointsAlongParallels.begin());
	itsAccumulatedPointsAlongParallels.front() = 0;
	//	itsData = himan::matrix<double>(itsAccumulatedPointsAlongParallels.back(), 1, 1, MissingDouble(),
	// MissingDouble());
}

point reduced_gaussian_grid::FirstPoint() const
{
	return LatLon(0);
}

point reduced_gaussian_grid::LastPoint() const
{
	return LatLon(Size());
}

std::vector<size_t> reduced_gaussian_grid::AccumulatedPointsAlongParallels() const
{
	return itsAccumulatedPointsAlongParallels;
}

std::vector<double> reduced_gaussian_grid::Latitudes() const
{
	return itsLatitudes;
}

std::ostream& reduced_gaussian_grid::Write(std::ostream& file) const
{
	grid::Write(file);

	file << "__itsN__ " << itsN << std::endl;
	file << "__itsNumberOfPointsAlongParallels__";

	for (auto& num : itsNumberOfPointsAlongParallels)
	{
		file << " " << num;
	}

	file << std::endl;

	return file;
}

point reduced_gaussian_grid::LatLon(size_t x, size_t y) const
{
	return point(static_cast<double>(x) * 360. / static_cast<double>(itsNumberOfPointsAlongParallels[y]),
	             itsLatitudes[y]);
}

point reduced_gaussian_grid::LatLon(size_t theLocationIndex) const
{
	auto it = std::upper_bound(itsAccumulatedPointsAlongParallels.begin(), itsAccumulatedPointsAlongParallels.end(),
	                           theLocationIndex);
	const size_t y = std::distance(itsAccumulatedPointsAlongParallels.begin(), --it);
	const size_t x = theLocationIndex - *it;

	return LatLon(x, y);
}

/*
double reduced_gaussian_grid::Value(size_t x, size_t y) const
{
    return itsData.At(itsAccumulatedPointsAlongParallels[y] + x);
}
*/
size_t reduced_gaussian_grid::LocationIndex(size_t x, size_t y) const
{
	return itsAccumulatedPointsAlongParallels[y] + x;
}

unique_ptr<grid> reduced_gaussian_grid::Clone() const
{
	return unique_ptr<grid>(new reduced_gaussian_grid(*this));
}

bool reduced_gaussian_grid::operator!=(const grid& other) const
{
	return !(other == *this);
}
bool reduced_gaussian_grid::operator==(const grid& other) const
{
	const reduced_gaussian_grid* g = dynamic_cast<const reduced_gaussian_grid*>(&other);

	if (g)
	{
		return EqualsTo(*g);
	}

	return false;
}

bool reduced_gaussian_grid::EqualsTo(const reduced_gaussian_grid& other) const
{
	if (!irregular_grid::EqualsTo(other))
	{
		return false;
	}

	if (itsN != other.N())
	{
		itsLogger.Trace("N does not match: " + to_string(itsN) + " vs " + to_string(other.N()));
		return false;
	}

	if (itsNumberOfPointsAlongParallels != other.NumberOfPointsAlongParallels())
	{
		return false;
	}

	if (itsAccumulatedPointsAlongParallels != other.AccumulatedPointsAlongParallels())
	{
		return false;
	}

	return true;
}

size_t reduced_gaussian_grid::Hash() const
{
	vector<size_t> hashes;
	hashes.push_back(Type());
	hashes.push_back(N());
	hashes.push_back(boost::hash_range(itsAccumulatedPointsAlongParallels.begin(),itsAccumulatedPointsAlongParallels.end()));
	return boost::hash_range(hashes.begin(),hashes.end());
}

std::map<int, std::vector<double>> reduced_gaussian_grid::cachedLatitudes;

std::vector<double> reduced_gaussian_grid::GetLatitudes(int theN)
{
	// mutex stuff required
	auto it = cachedLatitudes.find(theN);
	if (it != cachedLatitudes.end())
	{
		return it->second;
	}
	else
	{
		auto latitudes = numerical_functions::LegGauss<double>(static_cast<size_t>(2 * theN), false).first;
		std::for_each(latitudes.begin(), latitudes.end(), [](double& lat) { lat = std::asin(lat) * 180 / M_PI; });
		std::reverse(latitudes.begin(), latitudes.end());
		cachedLatitudes[theN] = latitudes;
		return latitudes;
	}
}
