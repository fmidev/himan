#include "interpolate.h"
#include "geoutil.h"
#include "lambert_conformal_grid.h"
#include "latitude_longitude_grid.h"
#include "numerical_functions.h"
#include "point_list.h"
#include "reduced_gaussian_grid.h"
#include "stereographic_grid.h"
#include "util.h"

#include "plugin_factory.h"
#include <Eigen/Dense>

#define HIMAN_AUXILIARY_INCLUDE

#include "querydata.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan;
using namespace Eigen;

// bool InsideTriangle(point a, point b, point c, point p);

namespace himan
{
namespace interpolate
{
bool IsSupportedGridForRotation(HPGridType type)
{
	switch (type)
	{
		case kRotatedLatitudeLongitude:
		case kStereographic:
		case kLambertConformalConic:
			return true;
		default:
			return false;
	}
}

bool InterpolateArea(const grid* baseGrid, info_t source)
{
	if (!source)
	{
		return false;
	}

#ifdef HAVE_CUDA
	if (source->PackedData()->HasData())
	{
		util::Unpack({source});
	}
#endif

	base<double> target;
	target.grid = std::shared_ptr<himan::grid>(baseGrid->Clone());

	if (baseGrid->Class() == kRegularGrid)
	{
		target.data.Resize(dynamic_cast<const regular_grid*>(baseGrid)->Ni(),
		                   dynamic_cast<const regular_grid*>(baseGrid)->Nj());
	}
	else if (baseGrid->Class() == kIrregularGrid)
	{
		target.data.Resize(baseGrid->Size(), 1);
	}

	if (interpolate::interpolator().Interpolate(*source->Base(), target, source->Param().InterpolationMethod()))
	{
		auto interpGrid = std::shared_ptr<grid>(baseGrid->Clone());

		interpGrid->AB(source->Grid()->AB());
		interpGrid->UVRelativeToGrid(source->Grid()->UVRelativeToGrid());

		source->Base()->grid = interpGrid;
		source->Base()->data = std::move(target.data);

		return true;
	}

	return false;
}

bool ReorderPoints(const grid* baseGrid, info_t info)
{
	if (!info)
	{
		return false;
	}

	// Worst case: cartesian product ie O(mn) ~ O(n^2)

	auto targetStations = dynamic_cast<const point_list*>(baseGrid)->Stations();
	auto sourceStations = std::dynamic_pointer_cast<point_list>(info->Grid())->Stations();
	const auto& sourceData = info->Data();
	matrix<double> newData(targetStations.size(), 1, 1, MissingDouble());

	if (targetStations.size() == 0 || sourceStations.size() == 0)
	{
		return false;
	}

	std::vector<station> newStations;

	for (size_t i = 0; i < targetStations.size(); i++)
	{
		station s1 = targetStations[i];

		bool found = false;

		for (size_t j = 0; j < sourceStations.size() && !found; j++)
		{
			station s2 = sourceStations[j];

			if (s1 == s2)
			{
				newStations.push_back(s1);
				newData.Set(i, sourceData.At(j));

				found = true;
			}
		}

		if (!found)
		{
			// itsLogger->Trace("Failed, source data does not contain all the same points as target");
			return false;
		}
	}

	std::dynamic_pointer_cast<point_list>(info->Grid())->Stations(newStations);

	auto b = info->Base();
	b->data = std::move(newData);

	return true;
}

bool Interpolate(const grid* baseGrid, std::vector<info_t>& infos, bool useCudaForInterpolation)
{
	for (const auto& info : infos)
	{
		bool needInterpolation = false;
		bool needPointReordering = false;

		/*
		 * Possible scenarios:
		 * 1. from regular to regular (basic area&grid interpolation)
		 * 2. from regular to irregular (area to point)
		 * 3. from irregular to irregular (limited functionality, basically just point reordering)
		 * 4. from irregular to regular, not supported, except if source is gaussian
		 */

		// 1.

		if (baseGrid->Class() == kRegularGrid && info->Grid()->Class() == kRegularGrid)
		{
			if (*baseGrid != *info->Grid())
			{
				needInterpolation = true;
			}

			// == operator does not test scanning mode !

			else if (dynamic_cast<const regular_grid*>(baseGrid)->ScanningMode() !=
			         std::dynamic_pointer_cast<regular_grid>(info->Grid())->ScanningMode())
			{
#ifdef HAVE_CUDA
				if (info->PackedData()->HasData())
				{
					// must unpack before swapping
					// itsLogger->Trace("Unpacking before swapping");
					util::Unpack({info});
				}
#endif
				util::Flip(info->Data());
				std::dynamic_pointer_cast<regular_grid>(info->Grid())
				    ->ScanningMode(dynamic_cast<const regular_grid*>(baseGrid)->ScanningMode());
			}
		}

		// 2.

		else if (baseGrid->Class() == kIrregularGrid && info->Grid()->Class() == kRegularGrid)
		{
			needInterpolation = true;
		}

		// 3.

		else if (baseGrid->Class() == kIrregularGrid && info->Grid()->Class() == kIrregularGrid)
		{
			if (*baseGrid != *info->Grid())
			{
				needPointReordering = true;
			}
		}

		// 4.

		else if (baseGrid->Class() == kRegularGrid && info->Grid()->Class() == kIrregularGrid)
		{
			if (info->Grid()->Type() == kReducedGaussian)
			{
				needInterpolation = true;
			}
			else
			{
				throw std::runtime_error("Unable to extrapolate from points to grid");
			}
		}

		if (needInterpolation && InterpolateArea(baseGrid, info) == false)
		{
			return false;
		}
		else if (needPointReordering && ReorderPoints(baseGrid, info))
		{
			return false;
		}
	}

	return true;
}

bool IsVectorComponent(const std::string& paramName)
{
	if (paramName == "U-MS" || paramName == "V-MS" || paramName == "WGU-MS" || paramName == "WGV-MS" ||
	    paramName == "IVELU-MS" || paramName == "IVELV-MS" || paramName == "WVELU-MS" || paramName == "WVELV-MS")
	{
		return true;
	}

	return false;
}

HPInterpolationMethod InterpolationMethod(const std::string& paramName, HPInterpolationMethod interpolationMethod)
{
	// Later we'll add this information to radon directly
	if (interpolationMethod == kBiLinear &&
	    (
	        // vector parameters
	        IsVectorComponent(paramName) || paramName == "DD-D" || paramName == "FF-MS" ||
	        // precipitation
	        paramName == "RR-KGM2" || paramName == "SNR-KGM2" || paramName == "GRI-KGM2" || paramName == "RRR-KGM2" ||
	        paramName == "RRRC-KGM2" || paramName == "RRRL-KGM2" || paramName == "SNRC-KGM2" ||
	        paramName == "SNRL-KGM2" || paramName == "RRRS-KGM2" || paramName == "RR-1-MM" || paramName == "RR-3-MM" ||
	        paramName == "RR-6-MM" || paramName == "RRI-KGM2" || paramName == "SNRI-KGM2" ||
	        paramName == "SNACC-KGM2" ||
	        // symbols
	        paramName == "CLDSYM-N" || paramName == "PRECFORM-N" || paramName == "PRECFORM2-N" ||
	        paramName == "FOGSYM-N" || paramName == "ICING-N" || paramName == "POTPRECT-N" ||
	        paramName == "POTPRECF-N" || paramName == "FOGINT-N" || paramName == "PRECTYPE-N"))
	{
#ifdef DEBUG
		std::cout << "Debug::interpolation Switching interpolation method from bilinear to nearest point" << std::endl;
#endif
		return kNearestPoint;  // nearest point in himan and newbase
	}

	return interpolationMethod;
}

void RotateVectorComponentsCPU(info<double>& UInfo, info<double>& VInfo)
{
	ASSERT(UInfo.Grid()->Type() == VInfo.Grid()->Type());

	auto& UVec = UInfo.Data().Values();
	auto& VVec = VInfo.Data().Values();

	ASSERT(UInfo.Grid()->Type() != kLatitudeLongitude);
	switch (UInfo.Grid()->Type())
	{
		case kRotatedLatitudeLongitude:
		{
			auto rll = std::dynamic_pointer_cast<rotated_latitude_longitude_grid>(UInfo.Grid());

			const point southPole = rll->SouthPole();

			for (size_t i = 0; i < UInfo.SizeLocations(); i++)
			{
				double U = UVec[i];
				double V = VVec[i];

				const point rotPoint = rll->RotatedLatLon(i);
				const point regPoint = rll->LatLon(i);
				const auto coeffs = util::EarthRelativeUVCoefficients(regPoint, rotPoint, southPole);

				double newU = std::get<0>(coeffs) * U + std::get<1>(coeffs) * V;
				double newV = std::get<2>(coeffs) * U + std::get<3>(coeffs) * V;

				UVec[i] = newU;
				VVec[i] = newV;
			}
		}
		break;

		case kLambertConformalConic:
		{
			auto lcc = std::dynamic_pointer_cast<lambert_conformal_grid>(UInfo.Grid());

			if (!lcc->UVRelativeToGrid())
			{
				return;
			}

			const double latin1 = lcc->StandardParallel1();
			const double latin2 = lcc->StandardParallel2();

			double cone;
			if (latin1 == latin2)
			{
				cone = sin(fabs(latin1) * constants::kDeg);
			}
			else
			{
				cone = (log(cos(latin1 * constants::kDeg)) - log(cos(latin2 * constants::kDeg))) /
				       (log(tan((90 - fabs(latin1)) * constants::kDeg * 0.5)) -
				        log(tan(90 - fabs(latin2)) * constants::kDeg * 0.5));
			}

			const double orientation = lcc->Orientation();

			for (UInfo.ResetLocation(); UInfo.NextLocation();)
			{
				size_t i = UInfo.LocationIndex();

				double U = UVec[i];
				double V = VVec[i];

				// http://www.mcs.anl.gov/~emconsta/wind_conversion.txt

				double londiff = UInfo.LatLon().X() - orientation;
				ASSERT(londiff >= -180 && londiff <= 180);
				ASSERT(UInfo.LatLon().Y() >= 0);

				const double angle = cone * londiff * constants::kDeg;
				double sinx, cosx;
				sincos(angle, &sinx, &cosx);

				UVec[i] = cosx * U + sinx * V;
				VVec[i] = -1 * sinx * U + cosx * V;
			}
		}
		break;

		case kStereographic:
		{
			// The same as lambert but with cone = 1

			auto sc = std::dynamic_pointer_cast<stereographic_grid>(UInfo.Grid());
			const double orientation = sc->Orientation();

			for (UInfo.ResetLocation(); UInfo.NextLocation();)
			{
				size_t i = UInfo.LocationIndex();

				double U = UVec[i];
				double V = VVec[i];

				const double angle = (UInfo.LatLon().X() - orientation) * constants::kDeg;
				double sinx, cosx;

				sincos(angle, &sinx, &cosx);

				UVec[i] = -1 * cosx * U + sinx * V;
				VVec[i] = -1 * -sinx * U + cosx * V;
			}
		}
		break;
		default:
			break;
	}
}

void RotateVectorComponents(info<double>& UInfo, info<double>& VInfo, bool useCuda)
{
	ASSERT(UInfo.Grid()->UVRelativeToGrid() == VInfo.Grid()->UVRelativeToGrid());

	if (!UInfo.Grid()->UVRelativeToGrid())
	{
		return;
	}

#ifdef HAVE_CUDA
	if (useCuda)
	{
		cudaStream_t stream;
		CUDA_CHECK(cudaStreamCreate(&stream));
		RotateVectorComponentsGPU(UInfo, VInfo, stream, 0, 0);
		CUDA_CHECK(cudaStreamSynchronize(stream));
	}
	else
#endif
	{
		RotateVectorComponentsCPU(UInfo, VInfo);
	}

	UInfo.Grid()->UVRelativeToGrid(false);
	VInfo.Grid()->UVRelativeToGrid(false);
}

std::pair<std::vector<size_t>, std::vector<double>> InterpolationWeights(reduced_gaussian_grid& source, point target)
{
	// target lon 0 <= lon < 360
	if (target.X() < 0.0)
		target.X(target.X() + 360.0);
	else if (target.X() >= 360.0)
		target.X(target.X() - 360.0);

	const auto lats = source.Latitudes();

	// check if point is inside domain
	if (target.Y() >= lats.front())
	{
		// point north of domain
		return std::make_pair(std::vector<size_t>{0}, std::vector<double>{MissingDouble()});
	}

	if (target.Y() <= lats.back())
	{
		// point south of domain
		return std::make_pair(std::vector<size_t>{0}, std::vector<double>{MissingDouble()});
	}

	// find y-indices
	auto south = std::lower_bound(lats.begin(), lats.end(), target.Y(), std::greater_equal<double>());
	auto north = south;
	north--;

	size_t y_north = static_cast<size_t>(std::distance(lats.begin(), north));
	size_t y_south = static_cast<size_t>(std::distance(lats.begin(), south));

	// find x-indices
	size_t x_north_west = static_cast<size_t>(
	    std::floor(static_cast<double>(source.NumberOfPointsAlongParallels()[y_north]) * target.X() / 360.));
	size_t x_north_east =
	    x_north_west < static_cast<size_t>(source.NumberOfPointsAlongParallels()[y_north] - 1) ? x_north_west + 1 : 0;
	size_t x_south_west = static_cast<size_t>(
	    std::floor(static_cast<double>(source.NumberOfPointsAlongParallels()[y_south]) * target.X() / 360.));
	size_t x_south_east =
	    x_south_west < static_cast<size_t>(source.NumberOfPointsAlongParallels()[y_south] - 1) ? x_south_west + 1 : 0;

	/*
	 *
	 *  a---------------b
	 *  |               |
	 *   \     p        |
	 *   |              |
	 *   c--------------d
	 *
	 *  Now we know the indices x|y of the four points a,b,c,d that surround p
	 */

	std::vector<size_t> idxs;
	idxs.reserve(4);
	idxs.push_back(source.LocationIndex(x_north_west, y_north));
	idxs.push_back(source.LocationIndex(x_north_east, y_north));
	idxs.push_back(source.LocationIndex(x_south_west, y_south));
	idxs.push_back(source.LocationIndex(x_south_east, y_south));

	// calculate weights by bilinear interpolation to point target (p) surrounded by points A|B|C|D
	const point& p = target;
	const point a = source.LatLon(idxs[0]);
	const point b = source.LatLon(idxs[1]);
	const point c = source.LatLon(idxs[2]);
	const point d = source.LatLon(idxs[3]);

	std::vector<double> weights(4);

	// Matrices and Vectors
	Eigen::Matrix4d A(4, 4);
	Eigen::MatrixXd xi(4, 2);
	Eigen::Vector4d x(4);
	Eigen::Vector2d phi(2);

	// Construct linear system of equations to be solved
	A(0, 0) = 1;
	A(1, 0) = 1;
	A(2, 0) = 1;
	A(3, 0) = 1;
	A(0, 1) = a.X();
	A(1, 1) = b.X() == 0.0 ? 360 : b.X();

	A(2, 1) = c.X();
	A(3, 1) = d.X() == 0.0 ? 360 : d.X();

	A(0, 2) = a.Y();
	A(1, 2) = b.Y();
	A(2, 2) = c.Y();
	A(3, 2) = d.Y();
	A(0, 3) = a.X() * a.Y();
	A(1, 3) = b.X() == 0.0 ? 360 * b.Y() : b.X() * b.Y();
	A(2, 3) = c.X() * c.Y();
	A(3, 3) = d.X() == 0.0 ? 360 * d.Y() : d.X() * d.Y();

	xi(0, 0) = 0;
	xi(1, 0) = 1;
	xi(2, 0) = 0;
	xi(3, 0) = 1;
	xi(0, 1) = 1;
	xi(1, 1) = 1;
	xi(2, 1) = 0;
	xi(3, 1) = 0;

	// Solve linear system
	xi = A.colPivHouseholderQr().solve(xi);

	x[0] = 1;
	x[1] = p.X();
	x[2] = p.Y();
	x[3] = p.X() * p.Y();

	xi.transposeInPlace();
	phi = xi * x;

	weights[0] = (1 - phi[0]) * phi[1];
	weights[1] = phi[0] * phi[1];
	weights[2] = (1 - phi[0]) * (1 - phi[1]);
	weights[3] = phi[0] * (1 - phi[1]);

	return std::make_pair(idxs, weights);
}

std::pair<std::vector<size_t>, std::vector<double>> InterpolationWeights(regular_grid& source, point target)
{
	auto xy = source.XY(target);

	if (IsMissing(xy.X()) || IsMissing(xy.Y()))
		return std::make_pair(std::vector<size_t>{0}, std::vector<double>{MissingDouble()});

	std::vector<size_t> idxs{static_cast<size_t>(xy.X()) + source.Ni() * static_cast<size_t>(xy.Y()),
	                         static_cast<size_t>(xy.X()) + source.Ni() * static_cast<size_t>(xy.Y()) + 1 -
	                             (static_cast<size_t>(xy.X()) == source.Ni() - 1 ? source.Ni() : 0),
	                         static_cast<size_t>(xy.X()) + source.Ni() * static_cast<size_t>(xy.Y() + 1),
	                         static_cast<size_t>(xy.X()) + source.Ni() * static_cast<size_t>(xy.Y() + 1) + 1 -
	                             (static_cast<size_t>(xy.X()) == source.Ni() - 1 ? source.Ni() : 0)};

	std::vector<double> weights(4);

	weights[0] = (1 - std::fmod(xy.X(), 1)) * (1 - std::fmod(xy.Y(), 1));
	weights[1] = std::fmod(xy.X(), 1) * (1 - std::fmod(xy.Y(), 1));
	weights[2] = (1 - std::fmod(xy.X(), 1)) * std::fmod(xy.Y(), 1);
	weights[3] = std::fmod(xy.X(), 1) * std::fmod(xy.Y(), 1);

	// Index is outside grid. This happens when target point is located on bottom edge
	// Set index to first grid point
	for (auto& idx : idxs)
	{
		if (idx > source.Size() - 1)
		{
			idx = 0;
		}
	}

	return std::make_pair(idxs, weights);
}

std::pair<size_t, double> NearestPoint(reduced_gaussian_grid& source, point target)
{
	// target lon 0 <= lon < 360
	if (target.X() < 0.0)
		target.X(target.X() + 360.0);
	else if (target.X() >= 360.0)
		target.X(target.X() - 360.0);

	const auto lats = source.Latitudes();

	// find y-indices
	auto south = std::lower_bound(lats.begin(), lats.end(), target.Y(), std::greater_equal<double>());
	auto north = south;
	north--;

	size_t y_north = std::distance(lats.begin(), north);
	size_t y_south = std::distance(lats.begin(), south);

	// find x-indices
	size_t x_north_west = static_cast<size_t>(
	    std::floor(static_cast<double>(source.NumberOfPointsAlongParallels()[y_north]) * target.X() / 360.));
	size_t x_north_east =
	    x_north_west < static_cast<size_t>(source.NumberOfPointsAlongParallels()[y_north] - 1) ? x_north_west + 1 : 0;
	size_t x_south_west = static_cast<size_t>(
	    std::floor(static_cast<double>(source.NumberOfPointsAlongParallels()[y_south]) * target.X() / 360.));
	size_t x_south_east =
	    x_south_west < static_cast<size_t>(source.NumberOfPointsAlongParallels()[y_south] - 1) ? x_south_west + 1 : 0;

	size_t nearest = x_north_west;
	for (auto p : {x_north_east, x_south_west, x_south_east})
	{
		if (geoutil::Distance(source.LatLon(p), target) < geoutil::Distance(source.LatLon(nearest), target))
			nearest = p;
	}

	return std::make_pair<size_t, double>(std::move(nearest), 1.0);
}

std::pair<size_t, double> NearestPoint(regular_grid& source, point target)
{
	auto xy = source.XY(target);
	if (IsMissing(xy.X()) || IsMissing(xy.Y()))
		return std::make_pair<size_t, double>(0, MissingDouble());

	// In case of point in wrap-around region on global grid
	if (std::round(xy.X()) == source.Ni())
		return std::make_pair<size_t, double>(source.Ni() * static_cast<size_t>(std::round(xy.Y())), 1.0);

	return std::make_pair<size_t, double>(
	    static_cast<size_t>(std::round(xy.X())) + source.Ni() * static_cast<size_t>(std::round(xy.Y())), 1.0);
}

// area_interpolation class member functions definitions
area_interpolation::area_interpolation(grid& source, grid& target, HPInterpolationMethod method)
    : itsInterpolation(target.Size(), source.Size())
{
	std::vector<Triplet<double>> coefficients;
	// compute weights in the interpolation matrix line by line, i.e. point by point on target grid
	for (size_t i = 0; i < target.Size(); ++i)
	{
		std::pair<std::vector<size_t>, std::vector<double>> w;
		switch (source.Type())
		{
			case kLatitudeLongitude:
			case kRotatedLatitudeLongitude:
			case kStereographic:
			case kLambertConformalConic:
				if (method == kBiLinear)
				{
					w = InterpolationWeights(dynamic_cast<regular_grid&>(source), target.LatLon(i));
				}
				else if (method == kNearestPoint)
				{
					auto np = NearestPoint(dynamic_cast<regular_grid&>(source), target.LatLon(i));
					w.first.push_back(np.first);
					w.second.push_back(np.second);
				}
				else
				{
					throw std::bad_typeid();
				}
				break;
			case kReducedGaussian:
				if (method == kBiLinear)
				{
					w = InterpolationWeights(dynamic_cast<reduced_gaussian_grid&>(source), target.LatLon(i));
				}
				else if (method == kNearestPoint)
				{
					auto np = NearestPoint(dynamic_cast<reduced_gaussian_grid&>(source), target.LatLon(i));
					w.first.push_back(np.first);
					w.second.push_back(np.second);
				}
				else
				{
					throw std::bad_typeid();
				}
				break;
			default:
				// what to throw?
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

void area_interpolation::Interpolate(base<double>& source, base<double>& target)
{
	Map<Matrix<double, Dynamic, Dynamic>> srcValues(source.data.ValuesAsPOD(), source.data.Size(), 1);
	Map<Matrix<double, Dynamic, Dynamic>> trgValues(target.data.ValuesAsPOD(), target.data.Size(), 1);

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

// Interpolator member functions
std::map<size_t, himan::interpolate::area_interpolation> interpolator::cache;
std::mutex interpolator::interpolatorAccessMutex;

bool interpolator::Insert(const base<double>& source, const base<double>& target, HPInterpolationMethod method)
{
	std::lock_guard<std::mutex> guard(interpolatorAccessMutex);

	std::pair<size_t, himan::interpolate::area_interpolation> insertValue;

	try
	{
		std::vector<size_t> hashes{method, source.grid->Hash(), target.grid->Hash()};
		insertValue.first = boost::hash_range(hashes.begin(), hashes.end());

		// area_interpolation is already present in cache
		if (cache.count(insertValue.first) > 0)
			return true;

		insertValue.second = himan::interpolate::area_interpolation(*source.grid, *target.grid, method);
	}
	catch (const std::exception& e)
	{
		return false;
	}

	return cache.insert(std::move(insertValue)).second;
}

bool interpolator::Interpolate(base<double>& source, base<double>& target, HPInterpolationMethod method)
{
	std::vector<size_t> hashes{method, source.grid->Hash(), target.grid->Hash()};
	auto it = cache.find(boost::hash_range(hashes.begin(), hashes.end()));

	if (it != cache.end())
	{
		it->second.Interpolate(source, target);
		return true;
	}

	else
	{
		Insert(source, target, method);
		return Interpolate(source, target, method);
	}
}

}  // namespace interpolate
}  // namespace himan
