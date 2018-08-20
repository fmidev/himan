/**
 * @file   area_interpolation.h
 *
 */

#ifndef AREA_INTERPOLATE_H
#define AREA_INTERPOLATE_H

#include "interpolate.h"
#include <Eigen/SparseCore>
#include <vector>
#include <mutex>

using namespace Eigen;

namespace himan
{
namespace interpolate
{
class area_interpolation
{
   public:
	area_interpolation() = default;
	area_interpolation(grid* source, grid* target);
	void Interpolate(grid* source, grid* target);
	size_t SourceSize() const;
	size_t TargetSize() const;
	std::string Identifier() const;

   private:
	std::string itsIdent;
	SparseMatrix<double, RowMajor> itsInterpolation;
};

// area_interpolation class member functions definitions
area_interpolation::area_interpolation(grid* source, grid* target)
    : itsIdent(source->Identifier() + target->Identifier()), itsInterpolation(target->Size(), source->Size())
{
	std::vector<Triplet<double>> coefficients;
	for (size_t i = 0; i < target->Size(); ++i)
	{
		std::pair<std::vector<size_t>, std::vector<double>> w;
		switch (source->Type())
		{
			case kReducedGaussian:
				w = InterpolationWeights(dynamic_cast<reduced_gaussian_grid*>(source), target->LatLon(i));
				break;
			default:
				throw std::bad_typeid();
				break;
		}

		for (size_t j = 0; j < w.first.size(); ++j)
		{
			coefficients.push_back(Triplet<double>(static_cast<int>(i), static_cast<int>(w.first[j]), w.second[j]));
		}
	}

	itsInterpolation.setFromTriplets(coefficients.begin(), coefficients.end());
}

void area_interpolation::Interpolate(grid* source, grid* target)
{
	Map<Matrix<double, Dynamic, Dynamic>> srcValues(source->Data().ValuesAsPOD(), source->Size(), 1);
	Map<Matrix<double, Dynamic, Dynamic>> trgValues(target->Data().ValuesAsPOD(), target->Size(), 1);

	trgValues = itsInterpolation * srcValues;
}

size_t area_interpolation::SourceSize() const
{
	return itsInterpolation.cols();
}

size_t area_interpolation::TargetSize() const
{
	return itsInterpolation.rows();
}

std::string area_interpolation::Identifier() const
{
	return itsIdent;
}


class interpolator
{
   public:
	static bool Insert(grid* source, grid* target);
	bool Interpolate(grid* source, grid* target);

   private:
	static std::mutex interpolatorAccessMutex;
	static std::map<std::string, interpolate::area_interpolation> cache;
};

std::map<std::string, himan::interpolate::area_interpolation> interpolator::cache;
std::mutex interpolator::interpolatorAccessMutex;

bool interpolator::Insert(grid* source, grid* target)
{
	std::lock_guard<std::mutex> guard(interpolatorAccessMutex);

	std::pair<std::string, himan::interpolate::area_interpolation> insertValue;

	try
	{
		insertValue.first = source->Identifier() + target->Identifier();
		insertValue.second = himan::interpolate::area_interpolation(source, target);
	}
	catch (const std::exception& e)
	{
		return false;
	}

	return cache.insert(std::move(insertValue)).second;
}

bool interpolator::Interpolate(grid* source, grid* target)
{
	auto it = cache.find(source->Identifier() + target->Identifier());

	if (it != cache.end())
	{
		it->second.Interpolate(source, target);
		return true;
	}
	return false;
}

}  // namespace interpolate
}  // namespace himan
#endif  // AREA_INTERPOLATE_H
