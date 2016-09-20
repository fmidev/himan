#include "cuda_plugin_helper.h"
#include "info_simple.h"
#include "interpolate.h"
#include "numerical_functions.h"
#include <thrust/sort.h>

const double kEpsilon = 1e-6;
using himan::kFloatMissing;

struct point
{
	double x;
	double y;

	__host__ __device__ point() : x(kFloatMissing), y(kFloatMissing) {}
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

__global__ void Print(double* __restrict__ arr, int i) { printf("%d %f\n", i, arr[i]); }
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
	double mode = kFloatMissing;

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

	double ret = kFloatMissing;

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
	double ret = kFloatMissing;

	// Find all four neighboring points

	point a(floor(gp.x), ceil(gp.y));
	point b(ceil(gp.x), ceil(gp.y));
	point c(floor(gp.x), floor(gp.y));
	point d(ceil(gp.x), floor(gp.y));

	// Assure neighboring points are inside grid and get values

	size_t size_x = sourceInfo.size_x;
	size_t size_y = sourceInfo.size_y;

	double av = kFloatMissing, bv = kFloatMissing, cv = kFloatMissing, dv = kFloatMissing;

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

	else if (av != kFloatMissing && bv != kFloatMissing && cv != kFloatMissing && dv != kFloatMissing)
	{
		ret = BiLinear(dist.x, dist.y, av, bv, cv, dv);
	}

	// x or y is at grid edge

	else if (fabs(dist.y) < kEpsilon && cv != kFloatMissing && dv != kFloatMissing)
	{
		ret = Linear(dist.x, cv, dv);
	}

	else if (fabs(dist.y - 1) < kEpsilon && av != kFloatMissing && bv != kFloatMissing)
	{
		ret = Linear(dist.x, av, bv);
	}

	else if (fabs(dist.x) < kEpsilon && cv != kFloatMissing && av != kFloatMissing)
	{
		ret = Linear(dist.y, cv, av);
	}

	else if (fabs(dist.x - 1) < kEpsilon && av != kFloatMissing && bv != kFloatMissing)
	{
		ret = Linear(dist.y, dv, bv);
	}

	// One point missing; these "triangulation" methods have been copied from NFmiInterpolation.cpp

	else if (av == kFloatMissing && bv != kFloatMissing && cv != kFloatMissing && dv != kFloatMissing)
	{
		double wsum = (dist.x * dist.y + (1 - dist.x) * (1 - dist.y) + dist.x * (1 - dist.y));

		ret = ((1 - dist.x) * (1 - dist.y) * cv + dist.x * (1 - dist.y) * dv + dist.x * dist.y * bv) / wsum;
	}
	else if (av != kFloatMissing && bv == kFloatMissing && cv != kFloatMissing && dv != kFloatMissing)
	{
		double wsum = ((1 - dist.x) * dist.y + (1 - dist.x) * (1 - dist.y) + dist.x * (1 - dist.y));

		ret = ((1 - dist.x) * (1 - dist.y) * cv + dist.x * (1 - dist.y) * dv + (1 - dist.x) * dist.y * av) / wsum;
	}
	else if (av != kFloatMissing && bv != kFloatMissing && cv == kFloatMissing && dv != kFloatMissing)
	{
		double wsum = ((1 - dist.x) * dist.y + dist.x * dist.y + dist.x * (1 - dist.y));

		ret = (dist.x * (1 - dist.y) * dv + (1 - dist.x) * dist.y * av + dist.x * dist.y * bv) / wsum;
	}
	else if (av != kFloatMissing && bv != kFloatMissing && cv != kFloatMissing && dv == kFloatMissing)
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

	if (ret == kFloatMissing)
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
	double ret = kFloatMissing;

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

	if (mode != kFloatMissing)
	{
		return mode;
	}

	double bilin = BiLinearInterpolation(d_source, sourceInfo, gp);

	arr[0] = fabs(av - bilin);
	arr[1] = fabs(bv - bilin);
	arr[2] = fabs(cv - bilin);
	arr[3] = fabs(dv - bilin);

	mode = Mode(arr);

	if (mode != kFloatMissing)
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

		double interp = kFloatMissing;

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

		assert(interp == interp);  // no NaN
		assert(interp < 1e30);     // No crazy values
	}
}

bool InterpolateCuda(himan::info& source, himan::info& base, himan::matrix<double>& targetData)
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

#ifdef DEBUG
	std::cout << "Debug::interpolate_cuda Interpolation method: " << (base.Param().InterpolationMethod()) << std::endl;
#endif

	// Determine all grid point coordinates that need to be interpolated.
	const size_t N = base.SizeLocations();

	::point* grid = new point[N];

	CreateGrid(source, base, grid);

	point* d_grid = 0;

	CUDA_CHECK(cudaMalloc((void**)&d_grid, sizeof(::point) * N));
	CUDA_CHECK(cudaMemcpyAsync(d_grid, grid, sizeof(::point) * N, cudaMemcpyHostToDevice, stream));

	double* d_source = 0;
	double* d_target = 0;

	CUDA_CHECK(cudaMalloc((void**)&d_source, source.SizeLocations() * sizeof(double)));
	CUDA_CHECK(cudaMalloc((void**)&d_target, N * sizeof(double)));

	auto sourceInfo = source.ToSimple();
	auto targetInfo = base.ToSimple();
	targetInfo->values = targetData.ValuesAsPOD();

	PrepareInfo(sourceInfo, d_source, stream);
	PrepareInfo(targetInfo);

#ifdef DEBUG
	memset(targetInfo->values, 0, N * sizeof(double));
#endif

	const int bs = 256;
	const int gs = N / bs + (N % bs == 0 ? 0 : 1);

	InterpolateCudaKernel<<<gs, bs, 0, stream>>>(d_source, d_target, d_grid, *sourceInfo, *targetInfo);

	delete[] grid;

	CUDA_CHECK(cudaStreamSynchronize(stream));

	himan::ReleaseInfo(sourceInfo);
	himan::ReleaseInfo(targetInfo, d_target, stream);

	CUDA_CHECK(cudaFree(d_source));
	CUDA_CHECK(cudaFree(d_target));
	CUDA_CHECK(cudaFree(d_grid));

	CUDA_CHECK(cudaStreamDestroy(stream));

	return true;
}
