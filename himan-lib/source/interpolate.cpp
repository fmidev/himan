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

#include "position.h"
#include <Eigen/Dense>

#define HIMAN_AUXILIARY_INCLUDE

#include "querydata.h"

#undef HIMAN_AUXILIARY_INCLUDE

#include "NFmiFastQueryInfo.h"

using namespace himan;

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

bool ToReducedGaussianCPU(info& base, info& source, matrix<double>& targetData)
{
	// switch to old MissingDouble() for compatibility with QD stuff
	targetData.MissingValue(kFloatMissing);

	auto q = GET_PLUGIN(querydata);

	std::shared_ptr<NFmiQueryData> sourceData = q->CreateQueryData(source, true);
	NFmiFastQueryInfo sourceInfo(sourceData.get());

	for (base.ResetLocation(); base.NextLocation();)
	{
		const point llpoint = base.LatLon();
		ASSERT(llpoint != point());

		const double value = sourceInfo.InterpolatedValue(NFmiPoint(llpoint.X(), llpoint.Y()));

		targetData.Set(base.LocationIndex(), value);
	}

	// back to himan
	targetData.MissingValue(MissingDouble());

	return true;
}

bool FromReducedGaussianCPU(info& base, info& source, matrix<double>& targetData)
{
	reduced_gaussian_grid* const gg = dynamic_cast<reduced_gaussian_grid*>(source.Grid());

	for (size_t i = 0; i < source.Grid()->Size(); i++)
	{
		const point p = source.Grid()->LatLon(i);
		auto w = InterpolationWeights(gg, p);

		double sum = 0;
		for (size_t j = 0; j < w.first.size(); ++j)
		{
			sum += w.second[j] * gg->Data().At(w.first[j]);
		}
		targetData.Set(i, sum);
	}

	return true;
}

bool InterpolateAreaCPU(info& base, info& source, matrix<double>& targetData)
{
#ifdef HAVE_CUDA

	if (source.Grid()->IsPackedData())
	{
		// We need to unpack
		util::Unpack({source.Grid()});
	}
#endif
	// switch to old MissingDouble() for compatibility with QD stuff
	targetData.MissingValue(kFloatMissing);

	auto q = GET_PLUGIN(querydata);

	std::shared_ptr<NFmiQueryData> baseData = q->CreateQueryData(base, true);
	std::shared_ptr<NFmiQueryData> sourceData = q->CreateQueryData(source, true);

	ASSERT(baseData);

	NFmiFastQueryInfo baseInfo = NFmiFastQueryInfo(baseData.get());
	NFmiFastQueryInfo sourceInfo(sourceData.get());

	auto param = std::string(sourceInfo.Param().GetParam()->GetName());

	int method = InterpolationMethod(param, base.Param().InterpolationMethod());
	sourceInfo.Param().GetParam()->InterpolationMethod(static_cast<FmiInterpolationMethod>(method));

	size_t i = 0;

	if (base.Grid()->Class() == kIrregularGrid)
	{
		for (baseInfo.ResetLocation(), i = 0; baseInfo.NextLocation(); i++)
		{
			double value = sourceInfo.InterpolatedValue(baseInfo.LatLon());
			targetData.Set(i, value);
		}
	}
	else
	{
		HPScanningMode mode = dynamic_cast<const regular_grid*>(base.Grid())->ScanningMode();

		if (mode == kBottomLeft)
		{
			baseInfo.Bottom();

			do
			{
				baseInfo.Left();

				do
				{
					double value = sourceInfo.InterpolatedValue(baseInfo.LatLon());

					targetData.Set(i, value);
					i++;
				} while (baseInfo.MoveRight());
			} while (baseInfo.MoveUp());
		}
		else if (mode == kTopLeft)
		{
			baseInfo.Top();

			do
			{
				baseInfo.Left();

				do
				{
					double value = sourceInfo.InterpolatedValue(baseInfo.LatLon());
					targetData.Set(i, value);
					i++;
				} while (baseInfo.MoveRight());
			} while (baseInfo.MoveDown());
		}
	}

	// back to Himan
	targetData.MissingValue(MissingDouble());

	return true;
}

bool InterpolateArea(info& target, info_t source, bool useCudaForInterpolation)
{
	if (!source)
	{
		return false;
	}

	HPGridType targetType = target.Grid()->Type();
	HPGridType sourceType = source->Grid()->Type();

	matrix<double> targetData(target.Data().SizeX(), target.Data().SizeY(), target.Data().SizeZ(),
	                          target.Data().MissingValue(), target.Data().MissingValue());

	switch (sourceType)
	{
		case kLatitudeLongitude:
		case kRotatedLatitudeLongitude:
		case kStereographic:
		case kLambertConformalConic:
		{
			switch (targetType)
			{
				case kLatitudeLongitude:
				case kRotatedLatitudeLongitude:
				case kStereographic:
				case kLambertConformalConic:
				case kPointList:
#ifdef HAVE_CUDA
					useCudaForInterpolation ? InterpolateAreaGPU(target, *source, targetData) :
#endif
					                        InterpolateAreaCPU(target, *source, targetData);
					break;

				case kReducedGaussian:
					ToReducedGaussianCPU(target, *source, targetData);
					break;

				default:
					throw std::runtime_error("Unsupported source grid type: " + HPGridTypeToString.at(targetType));
			}

			break;
		}

		case kReducedGaussian:
			switch (targetType)
			{
				case kLatitudeLongitude:
				case kRotatedLatitudeLongitude:
				case kStereographic:
					FromReducedGaussianCPU(target, *source, targetData);
					break;

				default:
					throw std::runtime_error("Unsupported target grid type for reduced gaussian: " +
					                         HPGridTypeToString.at(targetType));
			}
			break;
		default:
			throw std::runtime_error("Unsupported target grid type: " + HPGridTypeToString.at(sourceType));
	}

	auto interpGrid = std::shared_ptr<grid>(target.Grid()->Clone());
	interpGrid->AB(source->Grid()->AB());
	interpGrid->UVRelativeToGrid(source->Grid()->UVRelativeToGrid());

	interpGrid->Data(targetData);

	source->Grid(interpGrid);

	return true;
}

bool ReorderPoints(info& base, info_t info)
{
	if (!info)
	{
		return false;
	}

	// Worst case: cartesian product ie O(mn) ~ O(n^2)

	auto targetStations = dynamic_cast<point_list*>(base.Grid())->Stations();
	auto sourceStations = dynamic_cast<point_list*>(info->Grid())->Stations();
	auto sourceData = info->Grid()->Data();
	auto newData = matrix<double>(targetStations.size(), 1, 1, MissingDouble());

	if (targetStations.size() == 0 || sourceStations.size() == 0)
		return false;

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

	dynamic_cast<point_list*>(info->Grid())->Stations(newStations);
	info->Grid()->Data(newData);

	return true;
}

bool Interpolate(info& base, std::vector<info_t>& infos, bool useCudaForInterpolation)
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

		if (base.Grid()->Class() == kRegularGrid && info->Grid()->Class() == kRegularGrid)
		{
			if (*(base).Grid() != *info->Grid())
			{
				needInterpolation = true;
			}

			// == operator does not test scanning mode !

			else if (dynamic_cast<const regular_grid*>(base.Grid())->ScanningMode() !=
			         dynamic_cast<const regular_grid*>(info->Grid())->ScanningMode())
			{
#ifdef HAVE_CUDA
				if (info->Grid()->IsPackedData())
				{
					// must unpack before swapping
					// itsLogger->Trace("Unpacking before swapping");
					util::Unpack({info->Grid()});
				}
#endif
				dynamic_cast<regular_grid*>(info->Grid())
				    ->Swap(dynamic_cast<const regular_grid*>(base.Grid())->ScanningMode());
			}
		}

		// 2.

		else if (base.Grid()->Class() == kIrregularGrid && info->Grid()->Class() == kRegularGrid)
		{
			needInterpolation = true;
		}

		// 3.

		else if (base.Grid()->Class() == kIrregularGrid && info->Grid()->Class() == kIrregularGrid)
		{
			if (*base.Grid() != *info->Grid())
			{
				needPointReordering = true;
			}
		}

		// 4.

		else if (base.Grid()->Class() == kRegularGrid && info->Grid()->Class() == kIrregularGrid)
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

		if (needInterpolation && InterpolateArea(base, info, useCudaForInterpolation) == false)
		{
			return false;
		}
		else if (needPointReordering && ReorderPoints(base, info))
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

void RotateVectorComponentsCPU(info& UInfo, info& VInfo)
{
	ASSERT(UInfo.Grid()->Type() == VInfo.Grid()->Type());

	auto& UVec = UInfo.Data().Values();
	auto& VVec = VInfo.Data().Values();

	ASSERT(UInfo.Grid()->Type() != kLatitudeLongitude);
	switch (UInfo.Grid()->Type())
	{
		case kRotatedLatitudeLongitude:
		{
			rotated_latitude_longitude_grid* const rll = dynamic_cast<rotated_latitude_longitude_grid*>(UInfo.Grid());

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
			lambert_conformal_grid* const lcc = dynamic_cast<lambert_conformal_grid*>(UInfo.Grid());

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

			const stereographic_grid* sc = dynamic_cast<stereographic_grid*>(UInfo.Grid());
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

void RotateVectorComponents(info& UInfo, info& VInfo, bool useCuda)
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

std::pair<std::vector<size_t>, std::vector<double>> InterpolationWeights(reduced_gaussian_grid* source, point target)
{
	// target lon 0 <= lon < 360
	if (target.X() < 0.0)
		target.X(target.X() + 360.0);
	else if (target.X() >= 360.0)
		target.X(target.X() - 360.0);

	const auto lats = source->Latitudes();

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

	size_t y_north = static_cast<size_t> (std::distance(lats.begin(), north));
	size_t y_south = static_cast<size_t> (std::distance(lats.begin(), south));

	// find x-indices
	size_t x_north_west = static_cast<size_t>(
	    std::floor(static_cast<double>(source->NumberOfPointsAlongParallels()[y_north]) * target.X() / 360.));
	size_t x_north_east =
	    x_north_west < static_cast<size_t>(source->NumberOfPointsAlongParallels()[y_north] - 1) ? x_north_west + 1 : 0;
	size_t x_south_west = static_cast<size_t>(
	    std::floor(static_cast<double>(source->NumberOfPointsAlongParallels()[y_south]) * target.X() / 360.));
	size_t x_south_east =
	    x_south_west < static_cast<size_t>(source->NumberOfPointsAlongParallels()[y_south] - 1) ? x_south_west + 1 : 0;

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
	idxs.push_back(source->LocationIndex(x_north_west, y_north));
	idxs.push_back(source->LocationIndex(x_north_east, y_north));
	idxs.push_back(source->LocationIndex(x_south_west, y_south));
	idxs.push_back(source->LocationIndex(x_south_east, y_south));

	// calculate weights by bilinear interpolation to point target (p) surrounded by points A|B|C|D
	const point& p = target;
	const point a = source->LatLon(idxs[0]);
	const point b = source->LatLon(idxs[1]);
	const point c = source->LatLon(idxs[2]);
	const point d = source->LatLon(idxs[3]);

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

}  // namespace interpolate
}  // namespace himan
