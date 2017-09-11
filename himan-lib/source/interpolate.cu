#include "cuda_plugin_helper.h"
#include "info_simple.h"
#include "interpolate.h"
#include "numerical_functions.h"
#include <thrust/sort.h>

#include "stereographic_grid.h"

// these functions are defined in lambert_conformal_grid.cpp
extern double GetStandardParallel(himan::grid* g, int parallelno);
extern double GetOrientation(himan::grid* g);

const double kEpsilon = 1e-6;
using himan::IsMissingDouble;

struct point
{
	double x;
	double y;

	__host__ __device__ point() : x(himan::MissingDouble()), y(himan::MissingDouble()) {}
	__host__ __device__ point(double _x, double _y) : x(_x), y(_y) {}
};

__host__ __device__ unsigned int Index(unsigned int x, unsigned int y, unsigned int sx) { return y * sx + x; }
__host__ __device__ unsigned int Index(point p, unsigned int sx)
{
	return Index(static_cast<unsigned int>(p.x), static_cast<unsigned int>(p.y), sx);
}

__global__ void Swap(double* __restrict__ arr, size_t ni, size_t nj)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Flip with regards to x axis

	if (idx < nj * ni * 0.5)
	{
		const int i = fmod(static_cast<double>(idx), static_cast<double>(ni));
		const int j = floor(static_cast<double>(idx / ni));

		double upper = arr[idx];
		double lower = arr[Index(i, nj - 1 - j, ni)];

		arr[idx] = lower;
		arr[Index(i, nj - 1 - j, ni)] = upper;
	}
}

void CreateGrid(himan::info& sourceInfo, himan::info& targetInfo, ::point* grid)
{
	targetInfo.ResetLocation();

	int i = 0;

	while (targetInfo.NextLocation())
	{
		himan::point gp = sourceInfo.Grid()->XY(targetInfo.LatLon());

		grid[i].x = gp.X();
		grid[i].y = gp.Y();
		i++;
	}
}

__device__ double Mode(double* arr)
{
	thrust::sort(thrust::seq, arr, arr + 4);

	double num = arr[0];
	double mode = himan::MissingDouble();

	int count = 1;
	int modeCount = 0;

	bool multiModal = false;

	for (int i = 1; i < 4; i++)
	{
		double val = arr[i];

		if (fabs(val - num) < kEpsilon)
		{
			// increase occurrences for this number
			count++;

			if (count == modeCount)
			{
				multiModal = true;
			}
			else if (count > modeCount)
			{
				modeCount = count;
				mode = num;
				multiModal = false;
			}
		}
		else
		{
			// value changed
			count = 1;
			num = val;
		}
	}

	double ret = himan::MissingDouble();

	if (!multiModal)
	{
		ret = mode;
	}

	return ret;
}

__device__ bool IsInsideGrid(point& gp, size_t size_x, size_t size_y)
{
	// if interpolated grid points are negative, it means that we are outside the grid

	// sometime first grid point is -0, so we subtract a small value from first
	// grid point accept that value as well

	if (gp.x >= (0 - kEpsilon) && gp.y >= (0 - kEpsilon) &&

	    // if interpolated grid points are larger than source grid in x or y
	    // direction, it means again that we are outside of the area

	    ((fabs(gp.x - (size_x - 1)) < kEpsilon || __double2uint_ru(gp.x) < size_x) &&
	     (fabs(gp.y - (size_y - 1)) < kEpsilon || __double2uint_ru(gp.y) < size_y)))
	{
		return true;
	}

#ifdef EXTRADEBUG
	bool lc = gp.x >= (0 - kEpsilon) && gp.y >= (0 - kEpsilon);
	bool uc = (fabs(gp.x - (size_x - 1)) < kEpsilon || __double2uint_ru(gp.x) < size_x) &&
	          (fabs(gp.y - (size_y - 1)) < kEpsilon || __double2uint_ru(gp.y) < size_y);

	printf("gp x:%f y:%f discarded [%ld,%ld]: lower cond --> x:%d y:%d upper cond x:%d y:%d\n", gp.x, gp.y, size_x,
	       size_y, gp.x >= (0 - kEpsilon), gp.y >= (0 - kEpsilon), lc, uc);

#endif

	return false;
}

__device__ double NearestPointInterpolation(const double* __restrict__ d_source, himan::info_simple& sourceInfo,
                                            const point& gp)
{
	int rx = rint(gp.x);
	int ry = rint(gp.y);

	assert(rx >= 0 && rx <= sourceInfo.size_x);
	assert(ry >= 0 && ry <= sourceInfo.size_y);

	double npValue = d_source[Index(rx, ry, sourceInfo.size_x)];

	// Sometimes nearest point value is missing, but there is another point almost as close that
	// is not missing. Should we try to use that instead? This would mean that the interpolation
	// result would in some cases contain less missing values, but the cost is an extra branch
	// for *every* nearest point interpolation ever done!

	return npValue;
}

__device__ double BiLinearInterpolation(const double* __restrict__ d_source, himan::info_simple& sourceInfo,
                                        const point& gp)
{
	double ret = himan::MissingDouble();

	// Find all four neighboring points

	point a(floor(gp.x), ceil(gp.y));
	point b(ceil(gp.x), ceil(gp.y));
	point c(floor(gp.x), floor(gp.y));
	point d(ceil(gp.x), floor(gp.y));

	// Assure neighboring points are inside grid and get values

	size_t size_x = sourceInfo.size_x;
	size_t size_y = sourceInfo.size_y;

	double av = himan::MissingDouble(), bv = himan::MissingDouble(), cv = himan::MissingDouble(),
	       dv = himan::MissingDouble();

	if (IsInsideGrid(a, size_x, size_y))
	{
		av = d_source[Index(a, size_x)];
	}
	if (IsInsideGrid(b, size_x, size_y))
	{
		bv = d_source[Index(b, size_x)];
	}
	if (IsInsideGrid(c, size_x, size_y))
	{
		cv = d_source[Index(c, size_x)];
	}
	if (IsInsideGrid(d, size_x, size_y))
	{
		dv = d_source[Index(d, size_x)];
	}

	// Distance of interpolated point to neighboring points

	point dist(gp.x - c.x, gp.y - c.y);

	assert(dist.x >= 0 && dist.x <= 1);
	assert(dist.y >= 0 && dist.y <= 1);

	// If interpolated point is very close to source grid point, pick
	// the point value directly

	// This is preferred since nearest point is faster than bilinear, and
	// if wanted grid point =~ source grid point, the bilinear interpolation
	// value will be very close to nearest point value

	using namespace himan::numerical_functions::interpolation;

	if ((dist.x < kEpsilon || fabs(dist.x - 1) < kEpsilon) && (dist.y < kEpsilon || fabs(dist.y - 1) < kEpsilon))
	{
		ret = NearestPointInterpolation(d_source, sourceInfo, gp);
	}

	// All values present, regular bilinear interpolation

	else if (!IsMissingDouble(av) && !IsMissingDouble(bv) && !IsMissingDouble(cv) && !IsMissingDouble(dv))
	{
		ret = BiLinear(dist.x, dist.y, av, bv, cv, dv);
	}

	// x or y is at grid edge

	else if (fabs(dist.y) < kEpsilon && !IsMissingDouble(cv) && !IsMissingDouble(dv))
	{
		ret = Linear(dist.x, cv, dv);
	}

	else if (fabs(dist.y - 1) < kEpsilon && !IsMissingDouble(av) && !IsMissingDouble(bv))
	{
		ret = Linear(dist.x, av, bv);
	}

	else if (fabs(dist.x) < kEpsilon && !IsMissingDouble(cv) && !IsMissingDouble(av))
	{
		ret = Linear(dist.y, cv, av);
	}

	else if (fabs(dist.x - 1) < kEpsilon && !IsMissingDouble(av) && !IsMissingDouble(bv))
	{
		ret = Linear(dist.y, dv, bv);
	}

	// One point missing; these "triangulation" methods have been copied from NFmiInterpolation.cpp

	else if (IsMissingDouble(av) && !IsMissingDouble(bv) && !IsMissingDouble(cv) && !IsMissingDouble(dv))
	{
		double wsum = (dist.x * dist.y + (1 - dist.x) * (1 - dist.y) + dist.x * (1 - dist.y));

		ret = ((1 - dist.x) * (1 - dist.y) * cv + dist.x * (1 - dist.y) * dv + dist.x * dist.y * bv) / wsum;
	}
	else if (!IsMissingDouble(av) && IsMissingDouble(bv) && !IsMissingDouble(cv) && !IsMissingDouble(dv))
	{
		double wsum = ((1 - dist.x) * dist.y + (1 - dist.x) * (1 - dist.y) + dist.x * (1 - dist.y));

		ret = ((1 - dist.x) * (1 - dist.y) * cv + dist.x * (1 - dist.y) * dv + (1 - dist.x) * dist.y * av) / wsum;
	}
	else if (!IsMissingDouble(av) && !IsMissingDouble(bv) && IsMissingDouble(cv) && !IsMissingDouble(dv))
	{
		double wsum = ((1 - dist.x) * dist.y + dist.x * dist.y + dist.x * (1 - dist.y));

		ret = (dist.x * (1 - dist.y) * dv + (1 - dist.x) * dist.y * av + dist.x * dist.y * bv) / wsum;
	}
	else if (!IsMissingDouble(av) && !IsMissingDouble(bv) && !IsMissingDouble(cv) && IsMissingDouble(dv))
	{
		double wsum = ((1 - dist.x) * (1 - dist.y) + (1 - dist.x) * dist.y + dist.x * dist.y);

		ret = ((1 - dist.x) * (1 - dist.y) * cv + (1 - dist.x) * dist.y * av + dist.x * dist.y * bv) / wsum;
	}

#ifdef EXTRADEBUG
	else
	{
		printf("More than one point missing for gp x: %f y:%f --> a:%f b:%f c:%f d:%f | dx:%f dy:%f\n", gp.x, gp.y, av,
		       bv, cv, dv, dist.x, dist.y);
	}

	if ((ret != ret))
	{
		printf("gpx:%f gpy:%f [%ld %ld] |  dist x:%f y:%f\n", gp.x, gp.y, size_x, size_y, dist.x, dist.y);
		printf("av:%f bv:%f cv:%f dv:%f | interp:%f\n", av, bv, cv, dv, ret);
		printf("ax:%f ay:%f bx:%f by:%f cx:%f cy:%f dx:%f dy:%f\n", a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y);
		printf("is inside grid: a:%d b:%d c:%d d:%d\n", IsInsideGrid(a, size_x, size_y),
		       IsInsideGrid(b, size_x, size_y), IsInsideGrid(c, size_x, size_y), IsInsideGrid(d, size_x, size_y));
	}

#endif

	return ret;
}

__device__ double NearestPointValueInterpolation(const double* __restrict__ d_source, himan::info_simple& sourceInfo,
                                                 const point& gp)
{
	double ret = himan::MissingDouble();

	// Find all four neighboring points

	point a(floor(gp.x), ceil(gp.y));
	point b(ceil(gp.x), ceil(gp.y));
	point c(floor(gp.x), floor(gp.y));
	point d(ceil(gp.x), floor(gp.y));

	// Assure neighboring points are inside grid

	size_t size_x = sourceInfo.size_x;
	size_t size_y = sourceInfo.size_y;

	if (!IsInsideGrid(a, size_x, size_y) || !IsInsideGrid(b, size_x, size_y) || !IsInsideGrid(c, size_x, size_y) ||
	    !IsInsideGrid(d, size_x, size_y))
	{
		return ret;
	}

	// Neighbor values

	double av = d_source[Index(a, size_x)];
	double bv = d_source[Index(b, size_x)];
	double cv = d_source[Index(c, size_x)];
	double dv = d_source[Index(d, size_x)];

	// Find mode of neighboring points

	double arr[4] = {av, bv, cv, dv};
	double mode = Mode(arr);

	if (!IsMissingDouble(mode))
	{
		return mode;
	}

	double bilin = BiLinearInterpolation(d_source, sourceInfo, gp);

	arr[0] = fabs(av - bilin);
	arr[1] = fabs(bv - bilin);
	arr[2] = fabs(cv - bilin);
	arr[3] = fabs(dv - bilin);

	mode = Mode(arr);

	if (!IsMissingDouble(mode))
	{
		double min = fmin(arr[0], fmin(arr[1], fmin(arr[2], arr[3])));

		if (fabs(mode - min) < kEpsilon)
		{
			ret = bilin - mode;
		}
		else
		{
			ret = bilin - min;
		}
	}
	else
	{
		// no mode
		double min = fmin(arr[0], fmin(arr[1], fmin(arr[2], arr[3])));
		ret = bilin - min;
	}

	return ret;
}

__global__ void InterpolateCudaKernel(const double* __restrict__ d_source, double* __restrict__ d_target,
                                      const point* __restrict__ d_grid, himan::info_simple sourceInfo,
                                      himan::info_simple targetInfo)
{
	// idx is our pointer to the TARGET data in linear format

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < targetInfo.size_x * targetInfo.size_y)
	{
		// next we need to get x and y of the 'idx' in the source grid coordinates
		// to do that we first determine the i and j of the target grid coordinates

		const int i = fmod(static_cast<double>(idx), static_cast<double>(targetInfo.size_x));
		const int j = floor(static_cast<double>(idx / targetInfo.size_x));

		// with i and j we can get the grid point coordinates in the source grid

		point gp = d_grid[Index(i, j, targetInfo.size_x)];

		double interp = himan::MissingDouble();

		if (IsInsideGrid(gp, sourceInfo.size_x, sourceInfo.size_y))
		{
			// targetInfo.interpolation = himan::kNearestPointValue;

			switch (targetInfo.interpolation)
			{
				case himan::kBiLinear:
					interp = BiLinearInterpolation(d_source, sourceInfo, gp);
					break;

				case himan::kNearestPoint:
					interp = NearestPointInterpolation(d_source, sourceInfo, gp);
					break;

				case himan::kNearestPointValue:
					interp = NearestPointValueInterpolation(d_source, sourceInfo, gp);
					break;
			}
		}
#ifdef EXTRADEBUG
		else
		{
			printf("grid point x:%f y:%f discarded [%ld,%ld]\n", gp.x, gp.y, sourceInfo.size_x, sourceInfo.size_y);
		}
#endif
		d_target[idx] = interp;

		assert(interp == interp || IsMissingDouble(interp));  // no NaN
		assert(interp < 1e30 || IsMissingDouble(interp));     // No crazy values
	}
}

bool InterpolateAreaGPU(himan::info& base, himan::info& source, himan::matrix<double>& targetData)
{
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	if (base.Param().InterpolationMethod() == himan::kUnknownInterpolationMethod)
	{
		base.Param().InterpolationMethod(himan::kBiLinear);
	}
	else
	{
		auto newMethod =
		    himan::interpolate::InterpolationMethod(source.Param().Name(), base.Param().InterpolationMethod());
		auto newParam = base.Param();
		newParam.InterpolationMethod(newMethod);

		base.SetParam(newParam);
	}

	// Determine all grid point coordinates that need to be interpolated.
	const size_t N = base.SizeLocations();

	point* grid_ = 0;
	point* d_grid = 0;

	CUDA_CHECK(cudaMallocHost((void**)&grid_, N * sizeof(::point)));

	CreateGrid(source, base, grid_);

	CUDA_CHECK(cudaMalloc((void**)&d_grid, sizeof(::point) * N));
	CUDA_CHECK(cudaMemcpyAsync(d_grid, grid_, sizeof(::point) * N, cudaMemcpyHostToDevice, stream));

	double* d_source = 0;
	double* d_target = 0;

	CUDA_CHECK(cudaMalloc((void**)&d_source, source.SizeLocations() * sizeof(double)));
	CUDA_CHECK(cudaMalloc((void**)&d_target, N * sizeof(double)));

	auto sourceInfo = source.ToSimple();
	auto targetInfo = base.ToSimple();
	targetInfo->values = targetData.ValuesAsPOD();

	assert(targetInfo->values);

	PrepareInfo(sourceInfo, d_source, stream);
	PrepareInfo(targetInfo);

	const int bs = 256;
	const int gs = N / bs + (N % bs == 0 ? 0 : 1);

	InterpolateCudaKernel<<<gs, bs, 0, stream>>>(d_source, d_target, d_grid, *sourceInfo, *targetInfo);

	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaFreeHost(grid_));
	himan::ReleaseInfo(sourceInfo);

	himan::ReleaseInfo(targetInfo, d_target, stream);

	CUDA_CHECK(cudaFree(d_source));
	CUDA_CHECK(cudaFree(d_target));
	CUDA_CHECK(cudaFree(d_grid));

	CUDA_CHECK(cudaStreamDestroy(stream));

	return true;
}

__global__ void RotateLambert(double* __restrict__ d_u, double* __restrict__ d_v, const double* __restrict__ d_lon,
                              double cone, double orientation, himan::info_simple opts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.size_x * opts.size_y)
	{
		double U = d_u[idx];
		double V = d_v[idx];

		int i = fmod(static_cast<double>(idx), static_cast<double>(opts.size_x));
		int j = floor(static_cast<double>(idx / opts.size_x));

		double londiff = d_lon[idx] - orientation;
		const double angle = cone * londiff * himan::constants::kDeg;
		double sinx, cosx;
		sincos(angle, &sinx, &cosx);
		d_u[idx] = cosx * U + sinx * V;
		d_v[idx] = -1 * sinx * U + cosx * V;
	}
}

__global__ void RotateRotatedLatitudeLongitude(double* __restrict__ d_u, double* __restrict__ d_v,
                                               himan::info_simple opts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < opts.size_x * opts.size_y)
	{
		double U = d_u[idx];
		double V = d_v[idx];

		// Rotated to regular coordinates

		int i = fmod(static_cast<double>(idx), static_cast<double>(opts.size_x));  // idx - j * opts.size_x;
		int j = floor(static_cast<double>(idx / opts.size_x));

		double lon = opts.first_lon + i * opts.di;

		double lat = himan::MissingDouble();

		if (opts.j_scans_positive)
		{
			lat = opts.first_lat + j * opts.dj;
		}
		else
		{
			lat = opts.first_lat - j * opts.dj;
		}

		double SinYPole = sin((opts.south_pole_lat + 90.) * himan::constants::kDeg);
		double CosYPole = cos((opts.south_pole_lat + 90.) * himan::constants::kDeg);

		double SinXRot, CosXRot, SinYRot, CosYRot;

		sincos(lon * himan::constants::kDeg, &SinXRot, &CosXRot);
		sincos(lat * himan::constants::kDeg, &SinYRot, &CosYRot);

		double SinYReg = CosYPole * SinYRot + SinYPole * CosYRot * CosXRot;

		SinYReg = fmin(fmax(SinYReg, -1.), 1.);

		double YReg = asin(SinYReg) * himan::constants::kRad;

		double CosYReg = cos(YReg * himan::constants::kDeg);

		double CosXReg = (CosYPole * CosYRot * CosXRot - SinYPole * SinYRot) / CosYReg;

		CosXReg = fmin(fmax(CosXReg, -1.), 1.);
		double SinXReg = CosYRot * SinXRot / CosYReg;

		double XReg = acos(CosXReg) * himan::constants::kRad;

		if (SinXReg < 0.)
		{
			XReg = -XReg;
		}
		XReg += opts.south_pole_lon;

		// UV to earth relative

		double zxmxc = himan::constants::kDeg * (XReg - opts.south_pole_lon);

		double sinxmxc, cosxmxc;

		sincos(zxmxc, &sinxmxc, &cosxmxc);

		double PA = cosxmxc * CosXRot + CosYPole * sinxmxc * SinXRot;
		double PB = CosYPole * sinxmxc * CosXRot * SinYRot + SinYPole * sinxmxc * CosYRot - cosxmxc * SinXRot * SinYRot;
		double PC = (-SinYPole) * SinXRot / CosYReg;
		double PD = (CosYPole * CosYRot - SinYPole * CosXRot * SinYRot) / CosYReg;

		double newU = PA * U + PB * V;
		double newV = PC * U + PD * V;

		d_u[idx] = newU;
		d_v[idx] = newV;
	}
}

void RotateVectorComponentsGPU(himan::info& UInfo, himan::info& VInfo)
{
	const size_t N = UInfo.SizeLocations();
	const int bs = 256;
	const int gs = N / bs + (N % bs == 0 ? 0 : 1);

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	double* d_u = 0;
	double* d_v = 0;
	double* d_lon = 0;

	CUDA_CHECK(cudaMalloc((void**)&d_u, N * sizeof(double)));
	CUDA_CHECK(cudaMalloc((void**)&d_v, N * sizeof(double)));

	auto USimple = UInfo.ToSimple();
	auto VSimple = VInfo.ToSimple();

	PrepareInfo(USimple, d_u, stream);
	PrepareInfo(VSimple, d_v, stream);

	switch (UInfo.Grid()->Type())
	{
		case himan::kRotatedLatitudeLongitude:
			RotateRotatedLatitudeLongitude<<<gs, bs, 0, stream>>>(d_u, d_v, *USimple);
			break;

		case himan::kLambertConformalConic:
		{
			CUDA_CHECK(cudaMalloc((void**)&d_lon, N * sizeof(double)));

			double* lon = 0;

			CUDA_CHECK(cudaMallocHost((void**)&lon, N * sizeof(double)));

			for (UInfo.ResetLocation(); UInfo.NextLocation();)
			{
				lon[UInfo.LocationIndex()] = UInfo.LatLon().X();
			}

			CUDA_CHECK(cudaMemcpyAsync(d_lon, lon, N * sizeof(double), cudaMemcpyHostToDevice));

			const double latin1 = GetStandardParallel(UInfo.Grid(), 1);
			const double latin2 = GetStandardParallel(UInfo.Grid(), 2);
			const double orientation = GetOrientation(UInfo.Grid());

			assert(!himan::IsKHPMissingValue(latin1) && !himan::IsKHPMissingValue(orientation));
			double cone;

			using himan::constants::kDeg;

			if (latin1 == latin2)
			{
				cone = sin(fabs(latin1) * kDeg);
			}
			else
			{
				cone = (log(cos(latin1 * kDeg)) - log(cos(latin2 * kDeg))) /
				       (log(tan((90 - fabs(latin1)) * kDeg * 0.5)) - log(tan(90 - fabs(latin2)) * kDeg * 0.5));
			}

			RotateLambert<<<gs, bs, 0, stream>>>(d_u, d_v, d_lon, cone, orientation, *USimple);

			CUDA_CHECK(cudaStreamSynchronize(stream));
			CUDA_CHECK(cudaFreeHost(lon));
		}
		break;

		case himan::kStereographic:
		{
			const double orientation = dynamic_cast<himan::stereographic_grid*>(UInfo.Grid())->Orientation();
			CUDA_CHECK(cudaMalloc((void**)&d_lon, N * sizeof(double)));

			double* lon = 0;

			CUDA_CHECK(cudaMallocHost((void**)&lon, N * sizeof(double)));

			for (UInfo.ResetLocation(); UInfo.NextLocation();)
			{
				lon[UInfo.LocationIndex()] = UInfo.LatLon().X();
			}

			CUDA_CHECK(cudaMemcpyAsync(d_lon, lon, N * sizeof(double), cudaMemcpyHostToDevice));

			RotateLambert<<<gs, bs, 0, stream>>>(d_u, d_v, d_lon, 1, orientation, *USimple);

			CUDA_CHECK(cudaStreamSynchronize(stream));
			CUDA_CHECK(cudaFreeHost(lon));
		}
		break;

		default:
			break;
	}
	CUDA_CHECK(cudaStreamSynchronize(stream));

	himan::ReleaseInfo(USimple, d_u, stream);
	himan::ReleaseInfo(VSimple, d_v, stream);

	CUDA_CHECK(cudaFree(d_u));
	CUDA_CHECK(cudaFree(d_v));

	if (d_lon)
	{
		CUDA_CHECK(cudaFree(d_lon));
	}

	CUDA_CHECK(cudaStreamDestroy(stream));

	if (UInfo.Grid()->IsPackedData())
	{
		UInfo.Grid()->PackedData().Clear();
	}
	if (VInfo.Grid()->IsPackedData())
	{
		VInfo.Grid()->PackedData().Clear();
	}
}
