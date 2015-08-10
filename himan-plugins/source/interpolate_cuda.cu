#include "info_simple.h"
#include <NFmiLatLonArea.h>
#include <NFmiRotatedLatLonArea.h>
#include <NFmiStereographicArea.h>
#include <NFmiGrid.h>
#include "cuda_helper.h"
#include <thrust/sort.h>

const double kEpsilon = 1e-5;

struct point
{
	double x;
	double y;

	__host__ __device__
	point() : x(kFloatMissing), y(kFloatMissing) {}
	__host__ __device__
	point(double _x, double _y) : x(_x), y(_y) {}

};

__host__ __device__ 
int Index(int x, int y, int sx)
{
	return y * sx + x;
}

__host__ __device__ 
int Index(point p, int sx)
{
	return Index(static_cast<int> (p.x),static_cast<int> (p.y), sx);
}

__global__
void Flip(double* __restrict__ arr, size_t ni, size_t nj)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Flip with regards to x axis

	if (idx < nj * ni * 0.5)
	{
		size_t half = static_cast<size_t> (floor(static_cast<double>(nj/2)));

		const int i = fmod(static_cast<double> (idx), static_cast<double> (ni));
		const int j = floor(static_cast<double> (idx / ni));

		double upper = arr[idx];
		double lower = arr[Index(i,nj-1-j,ni)];

		arr[idx] = lower;
		arr[Index(i,nj-1-j,ni)] = upper;
		
	}
}

__global__ void Print(double* __restrict__ arr, int i) { printf("%d %f\n", i, arr[i]); }

 NFmiArea* CreateArea(himan::info_simple* info)
{
	NFmiPoint bl, tr;
	
	bl.X(info->first_lon);
	tr.X(bl.X() + (info->size_x - 1) * info->di);

	if (info->j_scans_positive)
	{	
		bl.Y(info->first_lat);
		tr.Y(bl.Y() + (info->size_y - 1) * info->dj);
	}
	else
	{
		tr.Y(info->first_lat);
		bl.Y(tr.Y() - (info->size_y - 1) * info->dj);
	}

	NFmiArea* area = 0;

	if (info->projection == himan::kLatLonProjection)
	{
		area = new NFmiLatLonArea(bl, tr);
	}
	else if (info->projection == himan::kRotatedLatLonProjection)
	{
		NFmiPoint sp(info->south_pole_lon, info->south_pole_lat);
		area = new NFmiRotatedLatLonArea(bl, tr, sp, NFmiPoint(0,0), NFmiPoint(1,1), true);
	}
	else if (info->projection == himan::kStereographicProjection)
	{
		area = new NFmiStereographicArea(bl, (info->size_x - 1) * info->di, (info->size_y - 1) * info->dj, info->orientation);
	}
	else
	{
		throw std::runtime_error("Invalid projection for cuda interpolation");
	}

	info->wraps_globally = area->PacificView();

	assert(area);
	return area;
}

point* CreateGrid(himan::info_simple* sourceInfo, himan::info_simple* targetInfo)
{

	NFmiArea* sourceArea = CreateArea(sourceInfo);
	NFmiArea* targetArea = CreateArea(targetInfo);
	
	NFmiGrid sourceGrid(sourceArea, sourceInfo->size_x, sourceInfo->size_y, kBottomLeft);
	NFmiGrid targetGrid(targetArea, targetInfo->size_x, targetInfo->size_y, kBottomLeft);
/*
	std::cout	<< "Source area BL: " << sourceArea->BottomLeftLatLon()
				<< "Source area TR: " << sourceArea->TopRightLatLon()
				<< "Source grid BL: " << sourceGrid.LatLonToGrid(sourceArea->BottomLeftLatLon())
				<< "Source grid TR: " << sourceGrid.LatLonToGrid(sourceArea->TopRightLatLon())
				<< "Source J scans positive: " << sourceInfo->j_scans_positive << std::endl
				<< "Target area BL: " << targetArea->BottomLeftLatLon()
				<< "Target area TR: " << targetArea->TopRightLatLon()
				<< "Target grid BL (relative): " << sourceGrid.LatLonToGrid(targetArea->BottomLeftLatLon())
				<< "Target grid TR (relative): " << sourceGrid.LatLonToGrid(targetArea->TopRightLatLon())
				<< "Target J scans positive: " << targetInfo->j_scans_positive << std::endl
				;
*/	
	delete (sourceArea);
	delete (targetArea);
	
	point* ret = new point[targetGrid.XNumber() * targetGrid.YNumber()];
	
	targetGrid.Reset();
	
	int i = 0;
	
	while(targetGrid.Next())
	{
		NFmiPoint gp = sourceGrid.LatLonToGrid(targetGrid.LatLon());
		
		ret[i].x = gp.X();
		ret[i].y = gp.Y();
		
		i++;
	}

	return ret;
}

__device__
double Mode(double* arr)
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

__device__
double Linear(double dx, double left, double right)
{
	// return (1 - dx) * left + dx * right;
	return fma(dx, right, fma(-dx, left, left)); 
}

__device__ 
double BiLinear(double dx, double dy, double a, double b, double c, double d)
{
	// Method below is faster but gives visible interpolation artifacts

	//double ab = Linear(dx, a, b);
	//double cd = Linear(dx, c, d);
	//return Linear(dy, ab, cd);

	// This one gives smooth interpolation surfaces
	return (1 - dx) * (1 - dy) * c + dx * (1 - dy) * d + (1 - dx) * dy * a + dx * dy * b;
}

__device__
double BiLinearInterpolation(const double* __restrict__ d_source, himan::info_simple& sourceInfo, const point& gp)
{
	// Find all four neighboring points

	point a(floor(gp.x), ceil(gp.y));
	point b(ceil(gp.x), ceil(gp.y));
	point c(floor(gp.x), floor(gp.y));
	point d(ceil(gp.x), floor(gp.y));

	// Neighbor values

	double av = d_source[Index(a,sourceInfo.size_x)];
	double bv = d_source[Index(b,sourceInfo.size_x)];
	double cv = d_source[Index(c,sourceInfo.size_x)];
	double dv = d_source[Index(d,sourceInfo.size_x)];	

	// Distance of interpolated point to neighboring points

	point dist(gp.x - c.x, gp.y - c.y);

	double ret = kFloatMissing;

	// TODO: Maybe add special cases if only one or two neighbors are missing?			
	
	if (av != kFloatMissing && bv != kFloatMissing && cv != kFloatMissing && dv != kFloatMissing)
	{
		ret = BiLinear(dist.x, dist.y, av, bv, cv, dv);
	}

#ifdef EXTRADEBUG
	// Neighbor point indexes in linear format

	int aidx = Index(a,sourceInfo.size_x);
	int bidx = Index(b,sourceInfo.size_x);
	int cidx = Index(c,sourceInfo.size_x);
	int didx = Index(d,sourceInfo.size_x);

	if (i == 0 && j == 0)
	{
		printf("x:%d y:%d gpx:%f gpy:%f\n", i, j, gp.x, gp.y);
		printf("a x:%d y:%d val:%f\n", int(a.x), int(a.y), av);
		printf("b x:%d y:%d val:%f\n", int(b.x), int(b.y), bv);
		printf("c x:%d y:%d val:%f\n", int(c.x), int(c.y), cv);
		printf("d x:%d y:%d val:%f\n", int(d.x), int(d.y), dv);
		printf("dist x:%f y:%f\n", dist.x, dist.y);
		printf("interp:%f\n", interp);
	}
#endif

	return ret;
}

__device__
double NearestPointInterpolation(const double* __restrict__ d_source, himan::info_simple& sourceInfo, const point& gp)
{
	int rx = rint(gp.x);
	int ry = rint(gp.y);

	assert(rx >= 0 && rx < sourceInfo.size_x);
	assert(ry >= 0 && ry < sourceInfo.size_y);

	return d_source[Index(rx,ry,sourceInfo.size_x)];

}

__device__
double NearestPointValueInterpolation(const double* __restrict__ d_source, himan::info_simple& sourceInfo, const point& gp)
{
	// Find all four neighboring points

	point a(floor(gp.x), ceil(gp.y));
	point b(ceil(gp.x), ceil(gp.y));
	point c(floor(gp.x), floor(gp.y));
	point d(ceil(gp.x), floor(gp.y));

	// Neighbor values

	double av = d_source[Index(a,sourceInfo.size_x)];
	double bv = d_source[Index(b,sourceInfo.size_x)];
	double cv = d_source[Index(c,sourceInfo.size_x)];
	double dv = d_source[Index(d,sourceInfo.size_x)];	

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
	
	double ret = kFloatMissing;
	
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


__global__ 
void InterpolateCudaKernel(const double* __restrict__ d_source, 
							double* __restrict__ d_target,
							const point* __restrict__ d_grid,
							himan::info_simple sourceInfo, 
							himan::info_simple targetInfo)
{

	// idx is our pointer to the TARGET data in linear format

	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < targetInfo.size_x * targetInfo.size_y)
	{
		// next we need to get x and y of the 'idx' in the source grid coordinates
		// to do that we first determine the i and j of the target grid coordinates
		
		const int i = fmod(static_cast<double> (idx), static_cast<double> (targetInfo.size_x));
		const int j = floor(static_cast<double> (idx / targetInfo.size_x));

		// with i and j we can get the grid point coordinates in the source grid
		
		point gp = d_grid[Index(i,j,targetInfo.size_x)];
		
		if (sourceInfo.wraps_globally && (gp.x < 0 || gp.x > sourceInfo.size_x - 1))
		{
			// wrap x if necessary
			// this might happen f.ex. with EC where grid start at 0 meridian and 
			// we interpolate from say -10 to 40 longitude

			while (gp.x < 0) gp.x += sourceInfo.size_x;
			while (gp.x > sourceInfo.size_x-1) gp.x -= sourceInfo.size_x-1;
		}
		
		double interp = kFloatMissing;
		
		if (
			// if interpolated grid points are negative, it means that we are outside
			// of the source area
				
			gp.x >= 0 && gp.y >= 0 &&
				
			// if interpolated grid points are larger than source grid in x or y
			// direction, it means again that we are outside of the area
			
			gp.x < (sourceInfo.size_x-1) && gp.y < (sourceInfo.size_y-1)
		)
		{
			
			//targetInfo.interpolation = himan::kNearestPointValue;

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

		d_target[idx] = interp ;

	}
}


bool InterpolateCuda(himan::info_simple* sourceInfo, himan::info_simple* targetInfo)
{
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	if (targetInfo->interpolation == himan::kUnknownInterpolationMethod)
	{
		targetInfo->interpolation = himan::kBiLinear;
	}

	// std::cout << "Interpolation method: " << targetInfo->interpolation << std::endl;
	
	/* Determine all grid point coordinates that need to be interpolated.
	 * This is done with newbase by explicitly looping through the grid.
	 * Initially I tried to implement it with just starting point and offset
	 * but the code was awkward and would not work with stereographic projections
	 * anyway.
	 */

	point* grid = CreateGrid(sourceInfo, targetInfo);
	
	const size_t N = targetInfo->size_x * targetInfo->size_y;
	
	point* d_grid = 0;
	CUDA_CHECK(cudaMalloc((void**) &d_grid, sizeof(point) * N));
	CUDA_CHECK(cudaMemcpyAsync(d_grid, grid, sizeof(point) * N, cudaMemcpyHostToDevice, stream));

	double* d_source = 0;
	double* d_target = 0;

	CUDA_CHECK(cudaMalloc((void **) &d_source, sourceInfo->size_x * sourceInfo->size_y * sizeof(double)));
	CUDA_CHECK(cudaMalloc((void **) &d_target, N * sizeof(double)));

#ifdef DEBUG
	CUDA_CHECK(cudaMemset(d_target, 0, targetInfo->size_x * targetInfo->size_y * 8));
#endif
	
	PrepareInfo(sourceInfo, d_source, stream);
	PrepareInfo(targetInfo);

	if (!sourceInfo->j_scans_positive)
	{
		// Force +x-y --> +x+y

		size_t N = sourceInfo->size_x * sourceInfo->size_y * 0.5 ;

		int bs = 256;
		int gs = N/bs + (N % bs == 0?0:1);
		Flip <<<gs,bs,0,stream>>>(d_source, sourceInfo->size_x, sourceInfo->size_y);

		sourceInfo->j_scans_positive = true;
	}
	
	const int bs = 256;
	const int gs = N/bs + (N % bs == 0?0:1);

	// Do bilinear transform on CUDA device
	InterpolateCudaKernel <<<gs,bs,0,stream>>>(d_source, d_target, d_grid, *sourceInfo, *targetInfo);
 
	CUDA_CHECK(cudaStreamSynchronize(stream));
	
	himan::ReleaseInfo(sourceInfo);
	himan::ReleaseInfo(targetInfo, d_target, stream);
	
	CUDA_CHECK(cudaFree(d_source));
	CUDA_CHECK(cudaFree(d_target));
	CUDA_CHECK(cudaFree(d_grid));
	
	CUDA_CHECK(cudaStreamDestroy(stream));
	
	return true;
}