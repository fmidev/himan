#include "interpolate.h"
#include "lambert_conformal_grid.h"
#include "latitude_longitude_grid.h"
#include "numerical_functions.h"
#include "point_list.h"
#include "reduced_gaussian_grid.h"
#include "stereographic_grid.h"
#include "util.h"

#include "plugin_factory.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "querydata.h"

#undef HIMAN_AUXILIARY_INCLUDE

#include "NFmiFastQueryInfo.h"

#ifdef HAVE_CUDA
extern bool InterpolateAreaGPU(himan::info& targetInfo, himan::info& baseInfo, himan::matrix<double>& targetData);
extern void RotateVectorComponentsGPU(himan::info& UInfo, himan::info& VInfo);
#endif

using namespace himan;

namespace himan
{
namespace interpolate
{
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
		assert(llpoint != point());

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

	std::vector<int> numOfPointsAlongParallels = gg->NumberOfPointsAlongParallels();

	double offset = 0;

	if (gg->TopLeft().X() == 0 && (gg->BottomRight().X() < 0 || gg->BottomRight().X() > 180))
	{
		offset = 360;
	}

	latitude_longitude_grid* baseGrid = 0;

	bool directInterpolation = false;

	if (base.Grid()->Type() == kLatitudeLongitude)
	{
		// FROM reduced gaussian TO latlon
		// Interpolated directly to target area without
		// going through regular interpolation path

		baseGrid = dynamic_cast<latitude_longitude_grid*>(base.Grid());
		directInterpolation = true;
	}
	else
	{
		// FROM reduced gaussian TO rotated latlon/stereographic
		// 1. Create a regular latitude longitude from reduced gaussian (match area coordinates)
		// 2. Pass this grid to regular "latlon --> rotlatlon/stereographic" interpolation routine

		auto tl = gg->TopLeft(), br = gg->BottomRight();
		point bl(tl.X(), br.Y()), tr(br.X(), tl.Y());

		if (tr.X() < 0)
		{
			tr.X(tr.X() + 360);
		}

		baseGrid = new latitude_longitude_grid(gg->ScanningMode(), bl, tr);

		double xlen = tr.X() - bl.X();
		if (xlen < 0) xlen += 360;

		const int numEquatorLongitudes = gg->NumberOfPointsAlongParallels()[gg->N()];

		baseGrid->Di(xlen / numEquatorLongitudes);
		baseGrid->Dj((tr.Y() - bl.Y()) / static_cast<double>(gg->Nj()));
		baseGrid->Nj(gg->Nj());
		baseGrid->Ni(numEquatorLongitudes);

		matrix<double> m(numEquatorLongitudes, gg->Nj(), 1, MissingDouble(), MissingDouble());
		baseGrid->Data(m);

		return false;
	}

	assert(baseGrid);
	assert(gg->TopLeft() != point());

	const double dj = (gg->TopLeft().Y() - gg->BottomRight().Y()) / (static_cast<double>(gg->Nj()) - 1.);
	assert(dj > 0);

	double lonspan =
	    (gg->BottomRight().X() - gg->TopLeft().X());  // longitude span of the whole gaussian area in degrees
	lonspan = (lonspan < 0) ? lonspan + 360 : lonspan;
	assert(lonspan >= 0 && lonspan <= 360);

	std::vector<double> result(baseGrid->Data().Size(), MissingDouble());

	HPInterpolationMethod interpolationMethod =
	    InterpolationMethod(source.Param().Name(), base.Param().InterpolationMethod());

	if (interpolationMethod == kNearestPoint)
	{
		for (size_t i = 0; i < baseGrid->Data().Size(); i++)
		{
			const auto llpoint = baseGrid->LatLon(i);  // latitude longitude point in target area
#if 0
			const double gg_y = (gg->TopLeft().Y() - llpoint.Y()) / dj; // gaussian grid y [0 .. Nj-1] (outside of grid interpolation is not allowed, even for small amounts of delta y)

			if (gg_y < 0 || gg_y > gg->Nj()-1) 
			{
				// lat outside gg grid
				targetData.Set(i, MissingDouble());
				continue;
			}

			const int np_y = static_cast<int> (rint(gg_y)); // nearest grid point in y direction

			const int numCurrentLongitudes = numOfPointsAlongParallels[static_cast<size_t> (np_y)]; // number of longitudes for the current parallel
			assert(numCurrentLongitudes > 0);

			const double di = (lonspan / (numCurrentLongitudes-1)); // longitude distance between two gaussian points in degrees for the current parallel

			assert(di > 0);
			double gg_x = (llpoint.X() - (gg->TopLeft().X() - offset)) / di; // gaussian grid x in current parallel

			if (offset != 0) gg_x = fmod(gg_x, numCurrentLongitudes-1); // wrap around if needed, do not allow any data to be from outside grid 

			assert(gg_x >= 0);

			if (gg_x < 0 || gg_x > numCurrentLongitudes-1) 
			{
				// lon outside gg grid
				targetData.Set(i, MissingDouble());
				continue;
			}
			const int np_x = static_cast<int> (rint(gg_x)); // nearest grid point in x direction in current parallel
			assert(np_x <= numCurrentLongitudes-1); // 0 ... numCurrentLongitudes-1

			const double interpValue = gg->Value(np_x, np_y);
#endif
			const point gpoint = gg->XY(llpoint);
			const point npoint = point(rint(gpoint.X()), rint(gpoint.Y()));

			// grid point coordinates are always positive
			assert(npoint.X() >= 0);
			assert(npoint.Y() >= 0);

			const double interpValue = gg->Value(static_cast<size_t>(npoint.X()), static_cast<size_t>(npoint.Y()));
			result[i] = interpValue;
		}
	}
	else if (interpolationMethod == kBiLinear)
	{
		for (size_t i = 0; i < baseGrid->Data().Size(); i++)
		{
			const auto llpoint = baseGrid->LatLon(i);  // latitude longitude point in target area

			/*
			 *     A -----+---------B
			 *     |  WD  |   WC    |
			 *  +--+------P---------+--------+
			 *  |         |                  |
			 *  |   WB    |        WA        |
			 *  |         |                  |
			 *  C---------+------------------D
			 *
			 * P = point of interest
			 * A,B,C,D = neighboring points
			 * WA,WB,WC,WD = weights for each neighboring point, area is inversely proportional to the weight
			 *
			 * Weights are calculated in latlon space.
			 */

			const double gg_y = (gg->TopLeft().Y() - llpoint.Y()) / dj;  // gaussian grid y [0 .. Nj-1] (outside of grid
			                                                             // interpolation is not allowed, even for small
			                                                             // amounts of delta y)

			if (gg_y < 0 || gg_y > static_cast<double>(gg->Nj()) - 1)
			{
				// lat outside gg grid
				targetData.Set(i, MissingDouble());
				continue;
			}

			const int numUpperLongitudes = numOfPointsAlongParallels[static_cast<size_t>(
			    static_cast<int>(floor(gg_y)))];  // number of longitudes for the upper parallel
			const int numLowerLongitudes = numOfPointsAlongParallels[static_cast<size_t>(
			    static_cast<int>(ceil(gg_y)))];  // number of longitudes for the lower parallel
			assert(numUpperLongitudes > 0 && numLowerLongitudes > 0);

			const double dlon_upper =
			    (lonspan / (numUpperLongitudes - 1));  // longitude distance between two points for the upper parallel
			const double dlon_lower =
			    (lonspan / (numLowerLongitudes - 1));  // longitude distance between two points for the lower parallel

			double a_lon = gg->TopLeft().X() +
			               floor((llpoint.X() - gg->TopLeft().X()) / dlon_upper) * dlon_upper;  // longitude for point a
			double b_lon = gg->TopLeft().X() + ceil((llpoint.X() - gg->TopLeft().X()) / dlon_upper) * dlon_upper;
			double c_lon = gg->TopLeft().X() + floor((llpoint.X() - gg->TopLeft().X()) / dlon_lower) * dlon_lower;
			double d_lon = gg->TopLeft().X() + ceil((llpoint.X() - gg->TopLeft().X()) / dlon_lower) * dlon_lower;

			if (offset != 0)
			{
				// wrap around for global grids
				if (a_lon < -180) a_lon += 360;
				if (b_lon < -180) b_lon += 360;
				if (c_lon < -180) c_lon += 360;
				if (d_lon < -180) d_lon += 360;
			}

			const double lat_upper =
			    gg->TopLeft().Y() - floor((gg->TopLeft().Y() - llpoint.Y()) / dj) * dj;  // latitude for a & b
			const double lat_lower =
			    gg->TopLeft().Y() - ceil((gg->TopLeft().Y() - llpoint.Y()) / dj) * dj;  // latitude for c & d

			const point a(a_lon, lat_upper);  // point a in latlon space
			const point b(b_lon, lat_upper);
			const point c(c_lon, lat_lower);
			const point d(d_lon, lat_lower);

			const point a_gp = gg->XY(a);  // point a in grid space
			const point b_gp = gg->XY(b);
			const point c_gp = gg->XY(b);
			const point d_gp = gg->XY(b);

			const double av = gg->Value(static_cast<size_t>(a_gp.X()), static_cast<size_t>(a_gp.Y()));  // point a value
			const double bv = gg->Value(static_cast<size_t>(b_gp.X()), static_cast<size_t>(b_gp.Y()));
			const double cv = gg->Value(static_cast<size_t>(c_gp.X()), static_cast<size_t>(c_gp.Y()));
			const double dv = gg->Value(static_cast<size_t>(d_gp.X()), static_cast<size_t>(d_gp.Y()));

			const double dlat_upper = fabs(lat_upper - llpoint.Y());  // latitude distance between a&b and p
			const double dlat_lower = fabs(llpoint.Y() - lat_lower);  // latitude distance between c&d and p

			double wa = dlat_lower * fabs(d.X() - llpoint.X());  // weight for area a
			double wb = dlat_lower * fabs(llpoint.X() - c.X());
			double wc = dlat_upper * fabs(b.X() - llpoint.X());
			double wd = dlat_upper * fabs(llpoint.X() - a.X());

			assert(wa >= 0 && std::isfinite(wa));
			assert(wb >= 0 && std::isfinite(wb));
			assert(wc >= 0 && std::isfinite(wc));
			assert(wd >= 0 && std::isfinite(wd));

			if (wa == 0 && wb == 0 && wc == 0 && wd == 0)
			{
				// Right at a longitude grid point. This happens at least when crossing
				// the equator: upper and lower parallels are symmetrical.
				// Do linear interpolation between latitude values.

				const double interpValue = numerical_functions::interpolation::Linear(gg_y, a.Y(), c.Y(), av, cv);
				result[i] = interpValue;
				continue;
			}

			// normalize weights

			const double wsum = 1 / (wa + wb + wc + wd);

			assert(wsum > 0.);

			wa *= wsum;
			wb *= wsum;
			wc *= wsum;
			wd *= wsum;

			const double interpValue = av * wa + bv * wb + cv * wc + dv * wd;
			assert(std::isfinite(interpValue));

			result[i] = interpValue;
		}
	}

	if (directInterpolation)
	{
		targetData.Set(result);
		return true;
	}

	auto newSource(source);
	baseGrid->Data().Set(result);
	newSource.Create(baseGrid);

	delete baseGrid;

	return InterpolateAreaCPU(base, newSource, targetData);
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

	q->UseDatabase(false);
	std::shared_ptr<NFmiQueryData> baseData = q->CreateQueryData(base, true);
	std::shared_ptr<NFmiQueryData> sourceData = q->CreateQueryData(source, true);

	assert(baseData);

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
		HPScanningMode mode = base.Grid()->ScanningMode();

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

bool InterpolateArea(info& target, std::vector<info_t> sources, bool useCudaForInterpolation)
{
	if (sources.size() == 0)
	{
		return false;
	}

	for (info_t& source : sources)
	{
		if (!source)
		{
			continue;
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
	}

	return true;
}

bool ReorderPoints(info& base, std::vector<info_t> infos)
{
	if (infos.size() == 0)
	{
		return false;
	}

	for (auto it = infos.begin(); it != infos.end(); ++it)
	{
		if (!(*it))
		{
			continue;
		}

		// Worst case: cartesian product ie O(mn) ~ O(n^2)

		auto targetStations = dynamic_cast<point_list*>(base.Grid())->Stations();
		auto sourceStations = dynamic_cast<point_list*>((*it)->Grid())->Stations();
		auto sourceData = (*it)->Grid()->Data();
		auto newData = matrix<double>(targetStations.size(), 1, 1, MissingDouble());

		if (targetStations.size() == 0 || sourceStations.size() == 0) return false;

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

		dynamic_cast<point_list*>((*it)->Grid())->Stations(newStations);
		(*it)->Grid()->Data(newData);
	}

	return true;
}

bool Interpolate(info& base, std::vector<info_t>& infos, bool useCudaForInterpolation)
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

	if (base.Grid()->Class() == kRegularGrid && infos[0]->Grid()->Class() == kRegularGrid)
	{
		if (*(base).Grid() != *(infos[0])->Grid())
		{
			needInterpolation = true;
		}

		// == operator does not test scanning mode !

		else if (base.Grid()->ScanningMode() != infos[0]->Grid()->ScanningMode())
		{
#ifdef HAVE_CUDA
			if (infos[0]->Grid()->IsPackedData())
			{
				// must unpack before swapping
				// itsLogger->Trace("Unpacking before swapping");
				util::Unpack({infos[0]->Grid()});
			}
#endif
			infos[0]->Grid()->Swap(base.Grid()->ScanningMode());
		}
	}

	// 2.

	else if (base.Grid()->Class() == kIrregularGrid && infos[0]->Grid()->Class() == kRegularGrid)
	{
		needInterpolation = true;
	}

	// 3.

	else if (base.Grid()->Class() == kIrregularGrid && infos[0]->Grid()->Class() == kIrregularGrid)
	{
		if (*base.Grid() != *infos[0]->Grid())
		{
			needPointReordering = true;
		}
	}

	// 4.

	else if (base.Grid()->Class() == kRegularGrid && infos[0]->Grid()->Class() == kIrregularGrid)
	{
		if (infos[0]->Grid()->Type() == kReducedGaussian)
		{
			needInterpolation = true;
		}
		else
		{
			throw std::runtime_error("Unable to extrapolate from points to grid");
		}
	}

	if (needInterpolation)
	{
		return InterpolateArea(base, infos, useCudaForInterpolation);
	}
	else if (needPointReordering)
	{
		return ReorderPoints(base, infos);
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
	        paramName == "FOGSYM-N" || paramName == "ICING-N"))
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
	assert(UInfo.Grid()->Type() == VInfo.Grid()->Type());

	auto& UVec = UInfo.Data().Values();
	auto& VVec = VInfo.Data().Values();

	assert(UInfo.Grid()->Type() != kLatitudeLongitude);
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
				assert(londiff >= -180 && londiff <= 180);
				assert(UInfo.LatLon().Y() >= 0);

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
	assert(UInfo.Grid()->UVRelativeToGrid() == VInfo.Grid()->UVRelativeToGrid());

	if (!UInfo.Grid()->UVRelativeToGrid())
	{
		return;
	}

#ifdef HAVE_CUDA
	if (useCuda)
	{
		RotateVectorComponentsGPU(UInfo, VInfo);
	}
	else
#endif
	{
		RotateVectorComponentsCPU(UInfo, VInfo);
	}

	UInfo.Grid()->UVRelativeToGrid(false);
	VInfo.Grid()->UVRelativeToGrid(false);
}

}  // namespace interpolate
}  // namespace himan
