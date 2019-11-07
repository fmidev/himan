#include "cuda_plugin_helper.h"
#include "interpolate.h"
#include "numerical_functions.h"
#include <thrust/sort.h>

#include "himan_common.h"
#include "latitude_longitude_grid.h"
#include "stereographic_grid.h"

// these functions are defined in lambert_conformal_grid.cpp
extern double GetOrientation(const himan::grid* g);
extern double GetCone(const himan::grid* g);

const double kEpsilon = 1e-6;
using himan::IsMissingDouble;

struct point
{
	double x;
	double y;

	__host__ __device__ point() : x(himan::MissingDouble()), y(himan::MissingDouble())
	{
	}
	__host__ __device__ point(double _x, double _y) : x(_x), y(_y)
	{
	}

	__host__ point(const himan::point& hp) : x(hp.X()), y(hp.Y())
	{
	}
};

__host__ __device__ unsigned int Index(unsigned int x, unsigned int y, unsigned int sx)
{
	return y * sx + x;
}
__host__ __device__ unsigned int Index(point p, unsigned int sx)
{
	if (himan::IsMissing(p.x) || himan::IsMissing(p.x))
		return 0;

	return Index(static_cast<unsigned int>(p.x) == sx ? 0 : static_cast<unsigned int>(p.x),
	             static_cast<unsigned int>(p.y), sx);
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

// Cache for grid points calculated by CreateGrid.
// In some cases this can save us a considerable amount of time (from 2300ms to 30ms).
struct cached_grid
{
	const std::string source;
	const std::string target;
	std::vector<point> points;

	__host__ cached_grid(const std::string& source, const std::string& target)
	    : source(source), target(target), points()
	{
	}
	__host__ cached_grid(const std::string& source, const std::string& target, const std::vector<point>& pts)
	    : source(source), target(target), points(pts)
	{
	}

	cached_grid(const cached_grid& other) = default;
	cached_grid& operator=(const cached_grid& other) = default;

	__host__ cached_grid& operator=(cached_grid&& other) noexcept = delete;
};

typedef std::lock_guard<std::mutex> lock_guard;

static std::vector<cached_grid> s_cachedGrids;
static std::mutex s_cachedGridMutex;

// This results in (for example): 'lcc_889_949', 'rll_1030_816'.
// Should be enough for uniquely determining a grid during one run.
static __host__ std::string CacheEntryName(const himan::info<double>& Info)
{
	std::stringstream ss;
	if (Info.Grid()->Class() == himan::kRegularGrid)
	{
		ss << himan::HPGridTypeToString.at(Info.Grid()->Type()) << "_"
		   << std::dynamic_pointer_cast<himan::regular_grid>(Info.Grid())->Ni() << "_"
		   << std::dynamic_pointer_cast<himan::regular_grid>(Info.Grid())->Nj();
	}
	else
	{
		ss << himan::HPGridTypeToString.at(Info.Grid()->Type()) << "_1_" << Info.Grid()->Size();
	}
	return ss.str();
}

void CreateGrid(himan::info<double>& sourceInfo, himan::info<double>& targetInfo, std::vector<::point>& grid)
{
	int i = 0;
	targetInfo.ResetLocation();

	const std::string sourceName = CacheEntryName(sourceInfo);
	const std::string targetName = CacheEntryName(targetInfo);

	std::vector<cached_grid>::iterator entry;
	std::vector<cached_grid>::iterator end;
	{
		lock_guard lock(s_cachedGridMutex);
		end = s_cachedGrids.end();
		entry = std::find_if(s_cachedGrids.begin(), s_cachedGrids.end(),
		                     [&](const cached_grid& g) { return sourceName == g.source && targetName == g.target; });
	}

	if (entry != end)
	{
		std::copy(entry->points.begin(), entry->points.end(), grid.begin());
	}
	else
	{
		if (sourceInfo.Grid()->Class() == himan::kRegularGrid)
		{
			while (targetInfo.NextLocation())
			{
				himan::point gp =
				    std::dynamic_pointer_cast<himan::regular_grid>(sourceInfo.Grid())->XY(targetInfo.LatLon());

				grid[i].x = gp.X();
				grid[i].y = gp.Y();
				i++;
			}
		}
		else
		{
			while (targetInfo.NextLocation())
			{
				grid[i].x = 1;
				grid[i].y = i + 1;
				i++;
			}
		}

		{
			lock_guard lock(s_cachedGridMutex);
			s_cachedGrids.emplace_back(sourceName, targetName, grid);
		}
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

template <typename T>
__global__ void RotateLambert(T* __restrict__ d_u, T* __restrict__ d_v, const double* __restrict__ d_lon, double cone,
                              double orientation, size_t size_x, size_t size_y)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size_x * size_y)
	{
		T U = d_u[idx];
		T V = d_v[idx];

		int i = fmod(static_cast<double>(idx), static_cast<double>(size_x));
		int j = floor(static_cast<double>(idx / size_x));

		double londiff = d_lon[idx] - orientation;
		const double angle = cone * londiff * himan::constants::kDeg;
		double sinx, cosx;
		sincos(angle, &sinx, &cosx);
		d_u[idx] = static_cast<T>(cosx * U + sinx * V);
		d_v[idx] = static_cast<T>(-1 * sinx * U + cosx * V);
	}
}

template <typename T>
__global__ void RotateRotatedLatitudeLongitude(T* __restrict__ d_u, T* __restrict__ d_v, size_t size_x, size_t size_y,
                                               point first, point south_pole, double di, double dj,
                                               himan::HPScanningMode mode)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size_x * size_y)
	{
		T U = d_u[idx];
		T V = d_v[idx];

		// Rotated to regular coordinates

		int i = fmod(static_cast<double>(idx), static_cast<double>(size_x));  // idx - j * opts.size_x;
		int j = floor(static_cast<double>(idx / size_x));

		double lon = first.x + i * di;

		double lat = himan::MissingDouble();

		if (mode == himan::kBottomLeft)
		{
			lat = first.y + j * dj;
		}
		else
		{
			lat = first.y - j * dj;
		}

		double SinYPole = sin((south_pole.y + 90.) * himan::constants::kDeg);
		double CosYPole = cos((south_pole.y + 90.) * himan::constants::kDeg);

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
		XReg += south_pole.x;

		// UV to earth relative

		double zxmxc = himan::constants::kDeg * (XReg - south_pole.x);

		double sinxmxc, cosxmxc;

		sincos(zxmxc, &sinxmxc, &cosxmxc);

		double PA = cosxmxc * CosXRot + CosYPole * sinxmxc * SinXRot;
		double PB = CosYPole * sinxmxc * CosXRot * SinYRot + SinYPole * sinxmxc * CosYRot - cosxmxc * SinXRot * SinYRot;
		double PC = (-SinYPole) * SinXRot / CosYReg;
		double PD = (CosYPole * CosYRot - SinYPole * CosXRot * SinYRot) / CosYReg;

		T newU = static_cast<T>(PA * U + PB * V);
		T newV = static_cast<T>(PC * U + PD * V);

		d_u[idx] = newU;
		d_v[idx] = newV;
	}
}

template <typename T>
void himan::interpolate::RotateVectorComponentsGPU(const grid* from, const grid* to, himan::matrix<T>& U,
                                                   himan::matrix<T>& V, cudaStream_t& stream, T* d_u, T* d_v)
{
	const size_t N = U.Size();
	const size_t memsize = N * sizeof(T);

	const int bs = 256;
	const int gs = N / bs + (N % bs == 0 ? 0 : 1);

	double* d_lon = nullptr;

	bool release = false;

	if (d_u == nullptr && d_v == nullptr)
	{
		release = true;
		CUDA_CHECK(cudaMalloc((void**)&d_u, memsize));
		CUDA_CHECK(cudaMalloc((void**)&d_v, memsize));

		CUDA_CHECK(cudaMemcpyAsync(d_u, U.ValuesAsPOD(), memsize, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpyAsync(d_v, V.ValuesAsPOD(), memsize, cudaMemcpyHostToDevice));
	}

	if (from->UVRelativeToGrid())
	{
		switch (from->Type())
		{
			case himan::kRotatedLatitudeLongitude:
			{
				const auto rg = dynamic_cast<const rotated_latitude_longitude_grid*>(from);
				RotateRotatedLatitudeLongitude<T>
				    <<<gs, bs, 0, stream>>>(d_u, d_v, U.SizeX(), U.SizeY(), ::point(rg->FirstPoint()),
				                            ::point(rg->SouthPole()), rg->Di(), rg->Dj(), rg->ScanningMode());
			}
			break;

			case himan::kLambertConformalConic:
			{
				CUDA_CHECK(cudaMalloc((void**)&d_lon, N * sizeof(double)));

				double* lon = nullptr;

				CUDA_CHECK(cudaMallocHost((void**)&lon, N * sizeof(double)));

				for (size_t i = 0; i < U.Size(); i++)
				{
					lon[i] = from->LatLon(i).X();
				}

				CUDA_CHECK(cudaMemcpyAsync(d_lon, lon, N * sizeof(double), cudaMemcpyHostToDevice));

				const double orientation = GetOrientation(from);
				const double cone = GetCone(from);

				RotateLambert<T><<<gs, bs, 0, stream>>>(d_u, d_v, d_lon, cone, orientation, U.SizeX(), U.SizeY());

				CUDA_CHECK(cudaStreamSynchronize(stream));
				CUDA_CHECK(cudaFreeHost(lon));
			}
			break;

			case himan::kStereographic:
			{
				const double orientation = dynamic_cast<const himan::stereographic_grid*>(from)->Orientation();
				CUDA_CHECK(cudaMalloc((void**)&d_lon, N * sizeof(double)));

				double* lon = nullptr;

				CUDA_CHECK(cudaMallocHost((void**)&lon, N * sizeof(double)));

				for (size_t i = 0; i < U.Size(); i++)
				{
					lon[i] = from->LatLon(i).X();
				}

				CUDA_CHECK(cudaMemcpyAsync(d_lon, lon, N * sizeof(double), cudaMemcpyHostToDevice));

				RotateLambert<T><<<gs, bs, 0, stream>>>(d_u, d_v, d_lon, 1, orientation, U.SizeX(), U.SizeY());

				CUDA_CHECK(cudaStreamSynchronize(stream));
				CUDA_CHECK(cudaFreeHost(lon));
			}
			break;

			default:
				break;
		}
	}

	CUDA_CHECK(cudaMemcpy(U.ValuesAsPOD(), d_u, memsize, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(V.ValuesAsPOD(), d_v, memsize, cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaStreamSynchronize(stream));

	if (release)
	{
		CUDA_CHECK(cudaFree(d_u));
		CUDA_CHECK(cudaFree(d_v));
	}

	if (d_lon)
	{
		CUDA_CHECK(cudaFree(d_lon));
	}
}

template void himan::interpolate::RotateVectorComponentsGPU<double>(const grid*, const grid*, himan::matrix<double>&,
                                                                    himan::matrix<double>&, cudaStream_t&, double* d_u,
                                                                    double* d_v);
template void himan::interpolate::RotateVectorComponentsGPU<float>(const grid*, const grid*, himan::matrix<float>&,
                                                                   himan::matrix<float>&, cudaStream_t&, float* d_u,
                                                                   float* d_v);
