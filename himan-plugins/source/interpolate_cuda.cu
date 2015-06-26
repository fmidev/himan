#include "info_simple.h"
#include <NFmiRotatedLatLonArea.h>
#include <NFmiLatLonArea.h>
#include <NFmiGrid.h>
#include "cuda_helper.h"

struct point
{
	double x;
	double y;

	__host__ __device__
	point() : x(kFloatMissing), y(kFloatMissing) {}
	__host__ __device__
	point(double _x, double _y) : x(_x), y(_y) {}

};

 NFmiArea* CreateArea(himan::info_simple* info)
{
	NFmiPoint bl(info->first_lon, info->first_lat);
	NFmiPoint tr(bl.X() + (info->size_x - 1) * info->di, bl.Y() + (info->size_y - 1) * info->dj);

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
	else
	{
		throw std::runtime_error("Invalid projection for cuda interpolation");
	}
	
	return area;
}

__host__ __device__ 
int index(int x, int y, int sx)
{
	return y * sx + x;
}

__host__ __device__ 
int index(point p, int sx)
{
	return index(static_cast<int> (p.x),static_cast<int> (p.y), sx);
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
	double ab = Linear(dx, a, b);
	double cd = Linear(dx, c, d);
	return Linear(dy, ab, cd);

	// return (1 - dx) * (1 - dy) * c + dx * (1 - dy) * d + (1 - dx) * dy * a + dx * dy * b;
}

__global__ 
void InterpolateCudaKernel(const double* __restrict__ source, 
							double* __restrict__ target, 
							himan::info_simple sourceInfo, 
							himan::info_simple targetInfo,
							point offset)
{

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (x < targetInfo.size_x && y < targetInfo.size_y)
	{
		point gp(offset.x + x * targetInfo.di / sourceInfo.di, offset.y + y * targetInfo.dj / sourceInfo.dj);
		
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
			// Find all four neighboring points

			point a(floor(gp.x), ceil(gp.y));
			point b(ceil(gp.x), ceil(gp.y));
			point c(floor(gp.x), floor(gp.y));
			point d(ceil(gp.x), floor(gp.y));

			// Neighbor values
			
			double av = source[index(a,sourceInfo.size_x)];
			double bv = source[index(b,sourceInfo.size_x)];
			double cv = source[index(c,sourceInfo.size_x)];
			double dv = source[index(d,sourceInfo.size_x)];

			// Distance of interpolated point to neighboring points
			
			point dist(gp.x - c.x, gp.y - c.y);

			if (av != kFloatMissing && bv != kFloatMissing && cv != kFloatMissing && dv != kFloatMissing)
			{
				interp = BiLinear(dist.x, dist.y, av, bv, cv, dv);
			}
			
			// TODO: Maybe add special cases if only one or two neighbors are missing?			

#ifdef DEBUG
			// Neighbor point indexes in linear format

			int aidx = index(a,sourceInfo.size_x);
			int bidx = index(b,sourceInfo.size_x);
			int cidx = index(c,sourceInfo.size_x);
			int didx = index(d,sourceInfo.size_x);

			/*
			if (interp < 100)
			{
				printf("x:%d y:%d gpx:%f gpy:%f\n", x, y, gp.x, gp.y);
				printf("a x:%d y:%d val:%f\n", int(a.x), int(a.y), av);
				printf("b x:%d y:%d val:%f\n", int(b.x), int(b.y), bv);
				printf("c x:%d y:%d val:%f\n", int(c.x), int(c.y), cv);
				printf("d x:%d y:%d val:%f\n", int(d.x), int(d.y), dv);
				printf("dist x:%f y:%f\n", dist.x, dist.y);
				printf("interp:%f\n", interp);
			} */
#endif
		}

		target[index(x, y, targetInfo.size_x)] = interp ;

	}
}

bool InterpolateCuda(himan::info_simple* sourceInfo, himan::info_simple* targetInfo)
{
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));
	
	NFmiArea* sourceArea = CreateArea(sourceInfo);
	NFmiArea* targetArea = CreateArea(targetInfo);
	
	//NFmiPoint interpbl(targetInfo->first_lon, targetInfo->first_lat);
	//NFmiPoint interptr(interpbl.X() + (targetInfo->size_x - 1) * targetInfo->di, interpbl.Y() + (targetInfo->size_y - 1) * targetInfo->dj);

	NFmiPoint interpbl = targetArea->BottomLeftLatLon();
	NFmiPoint interptr = targetArea->TopRightLatLon();
		
	NFmiGrid sourceGrid(sourceArea, sourceInfo->size_x, sourceInfo->size_y);

	NFmiPoint gpbl = sourceGrid.LatLonToGrid(interpbl);

#ifdef DEBUG
	NFmiPoint gptr = sourceGrid.LatLonToGrid(interptr);
#endif

	if (targetInfo->projection == himan::kRotatedLatLonProjection)
	{
		interpbl = dynamic_cast<NFmiRotatedLatLonArea*> (targetArea)->ToRotLatLon(interpbl);
		interptr = dynamic_cast<NFmiRotatedLatLonArea*> (targetArea)->ToRotLatLon(interptr);
	}
	
	point offset(gpbl.X(), gpbl.Y());

#ifdef DEBUG
	std::cout	<< "Source area BL: " << sourceArea->BottomLeftLatLon()
				<< "Source area TR: " << sourceArea->TopRightLatLon()
				<< "Source grid BL: " << sourceGrid.LatLonToGrid(sourceArea->BottomLeftLatLon())
				<< "Source grid TR: " << sourceGrid.LatLonToGrid(sourceArea->TopRightLatLon())
				<< "Target area BL: " << interpbl
				<< "Target area TR: " << interptr
				<< "Target grid BL: " << gpbl
				<< "Target grid TR: " << gptr
				<< "Source grid dx/dy " << sourceInfo->di << "/" << sourceInfo->di << std::endl
				<< "Target grid dx/dy " << targetInfo->di << "/" << targetInfo->dj << std::endl
				<< "Offset x/y: " << offset.x << "/" << offset.y << std::endl
				;
#endif	
	delete (sourceArea);
	delete (targetArea);

	double* d_source = 0;
	double* d_target = 0;

	CUDA_CHECK(cudaMalloc((void **) &d_source, sourceInfo->size_x * sourceInfo->size_y * sizeof(double)));
	CUDA_CHECK(cudaMalloc((void **) &d_target, targetInfo->size_x * targetInfo->size_y * sizeof(double)));
#ifdef DEBUG
	CUDA_CHECK(cudaMemset(d_target, 0, targetInfo->size_x * targetInfo->size_y * 8));
#endif
	
	PrepareInfo(sourceInfo, d_source, stream);
	PrepareInfo(targetInfo);

	const int blocksize = 16;
	const dim3 blocks(blocksize,blocksize);
	
	dim3 grids;
	
	grids.x = ceil((targetInfo->size_x+blocks.x-1)/blocks.x);
	grids.y = ceil((targetInfo->size_y+blocks.y-1)/blocks.y);

	// Do bilinear transform on CUDA device
	InterpolateCudaKernel <<<grids,blocks,0,stream>>>(d_source, d_target, *sourceInfo, *targetInfo, offset);
 
	CUDA_CHECK(cudaStreamSynchronize(stream));

	himan::ReleaseInfo(sourceInfo);
	himan::ReleaseInfo(targetInfo, d_target, stream);
	
	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaFree(d_source));
	CUDA_CHECK(cudaFree(d_target));

	CUDA_CHECK(cudaStreamDestroy(stream));

	return true;
}