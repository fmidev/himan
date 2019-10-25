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

namespace himan
{
namespace interpolate
{
// Function to return identifier for each supported datatype
// In the interpolation weight cache we store weights separately
// for each data type.

template <typename T>
int DataTypeId();

template <>
int DataTypeId<float>()
{
	return 1;
}

template <>
int DataTypeId<double>()
{
	return 2;
}

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

template <typename T>
bool InterpolateArea(const grid* baseGrid, std::shared_ptr<info<T>> source)
{
	if (!source)
	{
		return false;
	}

#ifdef HAVE_CUDA
	if (source->PackedData()->HasData())
	{
		util::Unpack<T>({source});
	}
#endif

	base<T> target;
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

	auto method = InterpolationMethod(source->Param().Name(), source->Param().InterpolationMethod());

	if (interpolate::interpolator<T>().Interpolate(*source->Base(), target, method))
	{
		auto interpGrid = std::shared_ptr<grid>(baseGrid->Clone());

		interpGrid->UVRelativeToGrid(source->Grid()->UVRelativeToGrid());

		source->Base()->grid = interpGrid;
		source->Base()->data = std::move(target.data);

		return true;
	}

	return false;
}

template bool InterpolateArea<double>(const grid*, std::shared_ptr<info<double>>);
template bool InterpolateArea<float>(const grid*, std::shared_ptr<info<float>>);

template <typename T>
bool ReorderPoints(const grid* baseGrid, std::shared_ptr<info<T>> info)
{
	if (!info)
	{
		return false;
	}

	// Worst case: cartesian product ie O(n^2)

	auto targetStations = dynamic_cast<const point_list*>(baseGrid)->Stations();
	auto sourceStations = std::dynamic_pointer_cast<point_list>(info->Grid())->Stations();
	const auto& sourceData = info->Data();
	matrix<T> newData(targetStations.size(), 1, 1, MissingValue<T>());

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

			if (s1.Id() == s2.Id())
			{
				newStations.push_back(s1);
				newData.Set(i, sourceData.At(j));

				found = true;
			}
		}

		if (!found)
		{
			// throw std::runtime_error("Failed, source data does not contain all the same points as target");
			return false;
		}
	}

	std::dynamic_pointer_cast<point_list>(info->Grid())->Stations(newStations);

	auto b = info->Base();
	b->data = std::move(newData);

	return true;
}

template bool ReorderPoints<double>(const grid*, std::shared_ptr<info<double>>);
template bool ReorderPoints<float>(const grid*, std::shared_ptr<info<float>>);

template <typename T>
bool Interpolate(const grid* baseGrid, std::vector<std::shared_ptr<info<T>>>& infos)
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
					util::Unpack<T>({info});
				}
#endif
				util::Flip<T>(info->Data());
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

		if (needInterpolation && InterpolateArea<T>(baseGrid, info) == false)
		{
			return false;
		}
		else if (needPointReordering && ReorderPoints<T>(baseGrid, info) == false)
		{
			return false;
		}
	}

	return true;
}

template bool Interpolate<double>(const grid*, std::vector<std::shared_ptr<info<double>>>&);
template bool Interpolate<float>(const grid*, std::vector<std::shared_ptr<info<float>>>&);

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

template <typename T>
void RotateVectorComponentsCPU(const grid* from, const grid* to, himan::matrix<T>& U, himan::matrix<T>& V)
{
	// First convert to earth relative

	logger log("interpolate");

	if (from->UVRelativeToGrid())
	{
		log.Trace("Rotating from " + HPGridTypeToString.at(from->Type()) + " to earth relative");

		switch (from->Type())  // source type
		{
			case kLatitudeLongitude:
				break;
			case kRotatedLatitudeLongitude:
			{
				const auto rll = dynamic_cast<const rotated_latitude_longitude_grid*>(from);
				point southPole = rll->SouthPole();

				if (southPole.Y() > 0)
				{
					southPole.Y(-southPole.Y());
					southPole.X(0);
				}

				for (size_t i = 0; i < U.Size(); i++)
				{
					T u = U[i];
					T v = V[i];

					const point rotPoint = rll->RotatedLatLon(i);
					const point regPoint = rll->LatLon(i);

					// Algorithm by J.E. HAUGEN (HIRLAM JUNE -92), modified by K. EEROLA
					// Algorithm originally defined in hilake/TURNDD.F

					const double southPoleY = constants::kDeg * (southPole.Y() + 90);

					double sinPoleY, cosPoleY;
					sincos(southPoleY, &sinPoleY, &cosPoleY);

					const double cosRegY = cos(constants::kDeg * regPoint.Y());  // zcyreg
					const double zxmxc = constants::kDeg * (regPoint.X() - southPole.X());

					double sinxmxc, cosxmxc;
					sincos(zxmxc, &sinxmxc, &cosxmxc);

					const double rotXRad = constants::kDeg * rotPoint.X();
					const double rotYRad = constants::kDeg * rotPoint.Y();

					double sinRotX, cosRotX;
					sincos(rotXRad, &sinRotX, &cosRotX);

					double sinRotY, cosRotY;
					sincos(rotYRad, &sinRotY, &cosRotY);

					const double PA = cosxmxc * cosRotX + cosPoleY * sinxmxc * sinRotX;
					const double PB = cosPoleY * sinxmxc * cosRotX * sinRotY + sinPoleY * sinxmxc * cosRotY -
					                  cosxmxc * sinRotX * sinRotY;
					const double PC = (-sinPoleY) * sinRotX / cosRegY;
					const double PD = (cosPoleY * cosRotY - sinPoleY * cosRotX * sinRotY) / cosRegY;

					U[i] = static_cast<T>(PA * u + PB * v);
					V[i] = static_cast<T>(PC * u + PD * v);
				}
			}
			break;

			case kLambertConformalConic:
			{
				auto lcc = dynamic_cast<const lambert_conformal_grid*>(from);
				const double cone = lcc->Cone();
				const double orientation = lcc->Orientation();

				for (size_t i = 0; i < U.Size(); i++)
				{
					T u = U[i];
					T v = V[i];

					// http://www.mcs.anl.gov/~emconsta/wind_conversion.txt

					double angle = from->LatLon(i).X() - orientation;
					ASSERT(angle >= -180 && angle <= 180);

					const double anglex = cone * angle * constants::kDeg;
					double sinx, cosx;
					sincos(anglex, &sinx, &cosx);

					U[i] = static_cast<T>(cosx * u + sinx * v);
					V[i] = static_cast<T>(-1 * sinx * u + cosx * v);
				}
			}
			break;

			case kStereographic:
			{
				// The same as lambert but with cone = 1

				const double orientation = dynamic_cast<const stereographic_grid*>(from)->Orientation();

				for (size_t i = 0; i < U.Size(); i++)
				{
					T u = U[i];
					T v = V[i];

					const double angle = (from->LatLon(i).X() - orientation) * constants::kDeg;
					double sinx, cosx;

					sincos(angle, &sinx, &cosx);

					U[i] = static_cast<T>(cosx * u + sinx * v);
					V[i] = static_cast<T>(-1 * sinx * u + cosx * v);
				}
			}
			break;
			default:
				throw std::runtime_error("Unable to rotate from " + HPGridTypeToString.at(from->Type()) + " to " +
				                         HPGridTypeToString.at(to->Type()));
		}
	}

	if (to->UVRelativeToGrid() == false)
	{
		log.Trace("Result grid has UVRelativeToGrid=false, no need for further rotation");
		return;
	}

	switch (to->Type())
	{
		case kLatitudeLongitude:
			break;

		case kRotatedLatitudeLongitude:
		{
			const auto rll = dynamic_cast<const rotated_latitude_longitude_grid*>(to);
			point southPole = rll->SouthPole();

			if (southPole.Y() > 0)
			{
				southPole.Y(-southPole.Y());
				southPole.X(0);
			}

			for (size_t i = 0; i < U.Size(); i++)
			{
				T u = U[i];
				T v = V[i];

				const point rotPoint = rll->RotatedLatLon(i);
				const point regPoint = rll->LatLon(i);

				// Algorithm by J.E. HAUGEN (HIRLAM JUNE -92), modified by K. EEROLA
				// Algorithm originally defined in hilake/TURNDD.F

				const double southPoleY = constants::kDeg * (southPole.Y() + 90);

				double sinPoleY, cosPoleY;
				sincos(southPoleY, &sinPoleY, &cosPoleY);

				const double sinRegY = sin(constants::kDeg * regPoint.Y());  // zsyreg
				const double cosRegY = cos(constants::kDeg * regPoint.Y());  // zcyreg

				double zxmxc = constants::kDeg * (regPoint.X() - southPole.X());

				double sinxmxc, cosxmxc;
				sincos(zxmxc, &sinxmxc, &cosxmxc);

				const double rotXRad = constants::kDeg * rotPoint.X();

				double sinRotX, cosRotX;
				sincos(rotXRad, &sinRotX, &cosRotX);

				const double cosRotY = cos(constants::kDeg * rotPoint.Y());  // zcyrot

				const double PA = cosPoleY * sinxmxc * sinRotX + cosxmxc * cosRotX;
				const double PB =
				    cosPoleY * cosxmxc * sinRegY * sinRotX - sinPoleY * cosRegY * sinRotX - sinxmxc * sinRegY * cosRotX;
				const double PC = sinPoleY * sinxmxc / cosRotY;
				const double PD = (sinPoleY * cosxmxc * sinRegY + cosPoleY * cosRegY) / cosRotY;

				U[i] = static_cast<T>(PA * u + PB * v);
				V[i] = static_cast<T>(PC * u + PD * v);
			}
		}
		break;

		case kLambertConformalConic:
		{
			auto lcc = dynamic_cast<const lambert_conformal_grid*>(to);
			const double cone = lcc->Cone();
			const double orientation = lcc->Orientation();

			for (size_t i = 0; i < U.Size(); i++)
			{
				T u = U[i];
				T v = V[i];

				// http://www.mcs.anl.gov/~emconsta/wind_conversion.txt

				const double angle = to->LatLon(i).X() - orientation;
				ASSERT(angle >= -180 && angle <= 180);

				const double anglex = cone * angle * constants::kDeg;
				double sinx, cosx;
				sincos(anglex, &sinx, &cosx);

				U[i] = static_cast<T>(cosx * u - sinx * v);
				V[i] = static_cast<T>(sinx * u + cosx * v);
			}
		}
		break;
		case kStereographic:
		{
			const double orientation = dynamic_cast<const stereographic_grid*>(to)->Orientation();

			for (size_t i = 0; i < U.Size(); i++)
			{
				T u = U[i];
				T v = V[i];

				const double angle = (to->LatLon(i).X() - orientation) * constants::kDeg;
				double sinx, cosx;

				sincos(angle, &sinx, &cosx);

				U[i] = static_cast<T>(cosx * u - sinx * v);
				V[i] = static_cast<T>(sinx * u + cosx * v);
			}
		}
		break;

		default:
			throw std::runtime_error("Unable to rotate from " + HPGridTypeToString.at(from->Type()) + " to " +
			                         HPGridTypeToString.at(to->Type()));
	}
}

template void RotateVectorComponentsCPU<double>(const grid*, const grid*, himan::matrix<double>&,
                                                himan::matrix<double>&);
template void RotateVectorComponentsCPU<float>(const grid*, const grid*, himan::matrix<float>&, himan::matrix<float>&);

template <typename T>
void RotateVectorComponents(const grid* from, const grid* to, himan::info<T>& UInfo, himan::info<T>& VInfo,
                            bool useCuda)
{
	ASSERT(UInfo.Grid()->UVRelativeToGrid() == VInfo.Grid()->UVRelativeToGrid());

	if (!UInfo.Grid()->UVRelativeToGrid())
	{
		return;
	}

#ifdef HAVE_CUDA
	if (useCuda)
	{
		if (UInfo.PackedData()->HasData() || VInfo.PackedData()->HasData())
		{
			throw std::runtime_error("Packed data needs to be unpacked before rotation on GPU");
		}

		cudaStream_t stream;
		CUDA_CHECK(cudaStreamCreate(&stream));
		RotateVectorComponentsGPU<T>(from, to, UInfo.Data(), VInfo.Data(), stream, 0, 0);
		CUDA_CHECK(cudaStreamSynchronize(stream));
	}
	else
#endif
	{
		RotateVectorComponentsCPU<T>(from, to, UInfo.Data(), VInfo.Data());
	}

	UInfo.Grid()->UVRelativeToGrid(false);
	VInfo.Grid()->UVRelativeToGrid(false);
}

template void RotateVectorComponents<double>(const grid*, const grid*, info<double>&, info<double>&, bool);
template void RotateVectorComponents<float>(const grid*, const grid*, info<float>&, info<float>&, bool);

template <typename T>
std::pair<std::vector<size_t>, std::vector<T>> InterpolationWeights(reduced_gaussian_grid& source, point target)
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
		return std::make_pair(std::vector<size_t>{0}, std::vector<T>{MissingValue<T>()});
	}

	if (target.Y() <= lats.back())
	{
		// point south of domain
		return std::make_pair(std::vector<size_t>{0}, std::vector<T>{MissingValue<T>()});
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

	std::vector<T> weights(4);

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

	weights[0] = static_cast<T>((1 - phi[0]) * phi[1]);
	weights[1] = static_cast<T>(phi[0] * phi[1]);
	weights[2] = static_cast<T>((1 - phi[0]) * (1 - phi[1]));
	weights[3] = static_cast<T>(phi[0] * (1 - phi[1]));

	return std::make_pair(idxs, weights);
}

template <typename T>
std::pair<std::vector<size_t>, std::vector<T>> InterpolationWeights(regular_grid& source, point target)
{
	auto xy = source.XY(target);

	if (IsMissing(xy.X()) || IsMissing(xy.Y()))
		return std::make_pair(std::vector<size_t>{0}, std::vector<T>{MissingValue<T>()});

	std::vector<size_t> idxs{static_cast<size_t>(xy.X()) + source.Ni() * static_cast<size_t>(xy.Y()),
	                         static_cast<size_t>(xy.X()) + source.Ni() * static_cast<size_t>(xy.Y()) + 1 -
	                             (static_cast<size_t>(xy.X()) == source.Ni() - 1 ? source.Ni() : 0),
	                         static_cast<size_t>(xy.X()) + source.Ni() * static_cast<size_t>(xy.Y() + 1),
	                         static_cast<size_t>(xy.X()) + source.Ni() * static_cast<size_t>(xy.Y() + 1) + 1 -
	                             (static_cast<size_t>(xy.X()) == source.Ni() - 1 ? source.Ni() : 0)};

	std::vector<T> weights(4);

	weights[0] = static_cast<T>((1 - std::fmod(xy.X(), 1)) * (1 - std::fmod(xy.Y(), 1)));
	weights[1] = static_cast<T>(std::fmod(xy.X(), 1) * (1 - std::fmod(xy.Y(), 1)));
	weights[2] = static_cast<T>((1 - std::fmod(xy.X(), 1)) * std::fmod(xy.Y(), 1));
	weights[3] = static_cast<T>(std::fmod(xy.X(), 1) * std::fmod(xy.Y(), 1));

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

template <typename T>
std::pair<size_t, T> NearestPoint(reduced_gaussian_grid& source, point target)
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

	return std::make_pair(nearest, 1.0);
}

template <typename T>
std::pair<size_t, T> NearestPoint(regular_grid& source, point target)
{
	auto xy = source.XY(target);
	if (IsMissing(xy.X()) || IsMissing(xy.Y()))
		return std::make_pair(0, MissingValue<T>());

	// In case of point in wrap-around region on global grid
	if (std::round(xy.X()) == source.Ni())
		return std::make_pair(source.Ni() * static_cast<size_t>(std::round(xy.Y())), 1.0);

	return std::make_pair(
	    static_cast<size_t>(std::round(xy.X())) + source.Ni() * static_cast<size_t>(std::round(xy.Y())), 1.0);
}

// area_interpolation class member functions definitions
template <typename T>
area_interpolation<T>::area_interpolation(grid& source, grid& target, HPInterpolationMethod method)
    : itsInterpolation(target.Size(), source.Size())
{
	std::vector<Triplet<T>> coefficients;
	// compute weights in the interpolation matrix line by line, i.e. point by point on target grid
	for (size_t i = 0; i < target.Size(); ++i)
	{
		std::pair<std::vector<size_t>, std::vector<T>> w;
		switch (source.Type())
		{
			case kLatitudeLongitude:
			case kRotatedLatitudeLongitude:
			case kStereographic:
			case kLambertConformalConic:
				if (method == kBiLinear)
				{
					w = InterpolationWeights<T>(dynamic_cast<regular_grid&>(source), target.LatLon(i));
				}
				else if (method == kNearestPoint)
				{
					auto np = NearestPoint<T>(dynamic_cast<regular_grid&>(source), target.LatLon(i));
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
					w = InterpolationWeights<T>(dynamic_cast<reduced_gaussian_grid&>(source), target.LatLon(i));
				}
				else if (method == kNearestPoint)
				{
					auto np = NearestPoint<T>(dynamic_cast<reduced_gaussian_grid&>(source), target.LatLon(i));
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
			coefficients.push_back(Triplet<T>(static_cast<int>(i), static_cast<int>(w.first[j]), w.second[j]));
		}
	}

	itsInterpolation.setFromTriplets(coefficients.begin(), coefficients.end());
}

template <typename T>
void area_interpolation<T>::Interpolate(base<T>& source, base<T>& target)
{
	Map<Matrix<T, Dynamic, Dynamic>> srcValues(source.data.ValuesAsPOD(), source.data.Size(), 1);
	Map<Matrix<T, Dynamic, Dynamic>> trgValues(target.data.ValuesAsPOD(), target.data.Size(), 1);

	trgValues = itsInterpolation * srcValues;
}

template <typename T>
size_t area_interpolation<T>::SourceSize() const
{
	return itsInterpolation.cols();
}

template <typename T>
size_t area_interpolation<T>::TargetSize() const
{
	return itsInterpolation.rows();
}

// Interpolator member functions
template <typename T>
std::map<size_t, area_interpolation<T>> interpolator<T>::cache;

template <typename T>
std::mutex interpolator<T>::interpolatorAccessMutex;

template <typename T>
bool interpolator<T>::Insert(const base<T>& source, const base<T>& target, HPInterpolationMethod method)
{
	std::lock_guard<std::mutex> guard(interpolatorAccessMutex);

	std::pair<size_t, himan::interpolate::area_interpolation<T>> insertValue;

	try
	{
		std::vector<size_t> hashes{method, source.grid->Hash(), target.grid->Hash()};
		insertValue.first = boost::hash_range(hashes.begin(), hashes.end());

		// area_interpolation is already present in cache
		if (cache.count(insertValue.first) > 0)
			return true;

		insertValue.second = himan::interpolate::area_interpolation<T>(*source.grid, *target.grid, method);
	}
	catch (const std::exception& e)
	{
		return false;
	}

	return cache.insert(std::move(insertValue)).second;
}

// template bool interpolator::Insert<double>(const base<double>&, const base<double>&, HPInterpolationMethod);
// template bool interpolator::Insert<float>(const base<float>&, const base<float>&, HPInterpolationMethod);

template <typename T>
bool interpolator<T>::Interpolate(base<T>& source, base<T>& target, HPInterpolationMethod method)
{
	std::vector<size_t> hashes{method, source.grid->Hash(), target.grid->Hash()};
	auto it = cache.find(boost::hash_range(hashes.begin(), hashes.end()));

	if (it != cache.end())
	{
		try
		{
			it->second.Interpolate(source, target);
			return true;
		}
		catch (const boost::bad_get& e)
		{
			std::cout << e.what() << std::endl;
			return false;
		}
	}

	else
	{
		Insert(source, target, method);
		return Interpolate(source, target, method);
	}
}

// template bool interpolator::Interpolate<double>(base<double>&, base<double>&, HPInterpolationMethod);
// template bool interpolator::Interpolate<float>(base<float>&, base<float>&, HPInterpolationMethod);

}  // namespace interpolate
}  // namespace himan
