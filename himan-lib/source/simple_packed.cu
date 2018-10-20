/**
 * @file simple_packed.cu
 *
 */

#include "cuda_helper.h"
#include "packed_data.h"
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

using namespace himan;

template <typename T>
bool IsHostPointer(T* ptr)
{
	cudaPointerAttributes attributes;
	cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

	bool ret;

	if (err == cudaErrorInvalidValue && ptr)
	{
		// memoery allocated with malloc
		ret = true;

		// Clear error buffer
		cudaGetLastError();
	}
	else if (err == cudaSuccess)
	{
		if (attributes.memoryType == cudaMemoryTypeHost)
		{
			ret = true;
		}
		else
		{
			ret = false;
		}
	}
	else
	{
		throw std::runtime_error("");
	}

	return ret;
}

__device__ void GetBitValue(unsigned char* p, long bitp, int* val)
{
	p += (bitp >> 3);
	*val = (*p & (1 << (7 - (bitp % 8))));
}

template <typename T>
__global__ void CopyWithBitmap(T* d_arr, int* d_b, T value, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		if (d_b[idx] == 1)
		{
			d_arr[idx] = value;
		}
	}
}

template <typename T>
__device__ void UnpackFullBytes(unsigned char* __restrict__ d_p, T* __restrict__ d_u, int* __restrict__ d_b,
                                packing_coefficients coeff, bool hasBitmap, int idx)
{
	int bc;
	unsigned long lvalue;

	int l = coeff.bitsPerValue / 8;

	int bm = idx;
	int value_found = 1;

	if (hasBitmap)
	{
		bm = d_b[idx];

		if (bm == 0)
		{
			d_u[idx] = MissingDouble();
			value_found = 0;
		}
		else
		{
			bm--;
		}
	}

	if (value_found)
	{
		size_t o = bm * l;

		lvalue = 0;
		lvalue <<= 8;
		lvalue |= d_p[o++];

		for (bc = 1; bc < l; bc++)
		{
			lvalue <<= 8;
			lvalue |= d_p[o++];
		}

		d_u[idx] = fma(lvalue, coeff.binaryScaleFactor, coeff.referenceValue) * coeff.decimalScaleFactor;
	}
}

template <typename T>
__device__ void UnpackUnevenBytes(unsigned char* __restrict__ d_p, T* __restrict__ d_u, int* __restrict__ d_b,
                                  packing_coefficients coeff, bool hasBitmap, int idx)
{
	int j = 0;
	unsigned long lvalue;

	int bm = idx;
	int value_found = 1;

	/*
	 * Check if bitmap is set.
	 * If bitmap is set and indicates that value for this element is missing, do
	 * not proceed to calculating phase.
	 *
	 * If bitmap is set and indicates that value exists for this element, the index
	 * for the actual data is the one indicated by the bitmap array. From this index
	 * we reduce one (1) because that one is added to the value in unpack_bitmap.
	 */

	if (hasBitmap)
	{
		bm = d_b[idx];

		if (bm == 0)
		{
			d_u[idx] = MissingDouble();
			value_found = 0;
		}
		else
		{
			bm--;
		}
	}

	if (value_found)
	{
		long bitp = coeff.bitsPerValue * bm;

		lvalue = 0;

		for (j = 0; j < coeff.bitsPerValue; j++)
		{
			lvalue <<= 1;
			int val;

			GetBitValue(d_p, bitp, &val);

			if (val)
				lvalue += 1;

			bitp += 1;
		}

		d_u[idx] = fma(lvalue, coeff.binaryScaleFactor, coeff.referenceValue) * coeff.decimalScaleFactor;
	}
}

template <typename T>
__global__ void UnpackKernel(unsigned char* d_p, T* d_u, int* d_b, packing_coefficients coeff, bool hasBitmap, size_t N)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		if (coeff.bitsPerValue %
		    8)  // modulo is expensive but "Compiler will convert literal power-of-2 divides to bitwise shifts"
		{
			UnpackUnevenBytes<T>(d_p, d_u, d_b, coeff, hasBitmap, idx);
		}
		else
		{
			UnpackFullBytes<T>(d_p, d_u, d_b, coeff, hasBitmap, idx);
		}
	}
}

template <typename T>
__host__ void FillStaticGrid(const simple_packed* src, T* dst, cudaStream_t* stream)
{
	// For empty grid (all values missing), grib_api gives reference value 1!
	const double fillValue = src->coefficients.referenceValue;
	const size_t N = src->unpackedLength;

	const int blockSize = 512;
	const int gridSize = N / blockSize + (N % blockSize == 0 ? 0 : 1);

	if (src->HasBitmap())
	{
		if (IsHostPointer(dst))
		{
			for (size_t i = 0; i < N; i++)
			{
				if (src->bitmap[i])
				{
					dst[i] = static_cast<T>(fillValue);
				}
			}
		}
		else
		{
			int* d_b = 0;
			CUDA_CHECK(cudaMalloc((void**)(&d_b), src->bitmapLength * sizeof(int)));
			CUDA_CHECK(
			    cudaMemcpyAsync(d_b, src->bitmap, src->bitmapLength * sizeof(int), cudaMemcpyHostToDevice, *stream));

			CopyWithBitmap<T><<<blockSize, gridSize, 0, stream>>>(dst, d_b, fillValue, N);

			CUDA_CHECK(cudaStreamSynchronize(*stream));
			CUDA_CHECK(cudaFree(d_b));
		}
	}
	else
	{
		if (IsHostPointer(dst))
		{
			std::fill(dst, dst + N, fillValue);
		}
		else
		{
			thrust::device_ptr<T> ptr = thrust::device_pointer_cast(dst);
			thrust::fill(ptr, ptr + N, fillValue);
		}
	}
}

template <typename T>
__host__ void packing::Unpack(const simple_packed* src, T* dst, cudaStream_t* stream)
{
	ASSERT(dst);

	const size_t N = src->unpackedLength;

	ASSERT(N > 0);

	const int blockSize = 512;
	const int gridSize = src->unpackedLength / blockSize + (src->unpackedLength % blockSize == 0 ? 0 : 1);

	bool destroyStream = false;

	if (!stream)
	{
		stream = new cudaStream_t;
		CUDA_CHECK(cudaStreamCreate(stream));
		destroyStream = true;
	}

	if (src->packedLength == 0 && src->coefficients.bitsPerValue == 0)
	{
		// Special case for static grid
		FillStaticGrid<T>(src, dst, stream);

		if (destroyStream)
		{
			CUDA_CHECK(cudaStreamDestroy(*stream));
		}

		return;
	}

	unsigned char* d_p = 0;  // device-packed data

	CUDA_CHECK(cudaMalloc((void**)(&d_p), src->packedLength * sizeof(unsigned char)));
	CUDA_CHECK(
	    cudaMemcpyAsync(d_p, src->data, src->packedLength * sizeof(unsigned char), cudaMemcpyHostToDevice, *stream));

	// Allocate memory on device for unpacked data

	T* d_dst = 0;

	bool releaseUnpackedDeviceMemory = false;

	if (IsHostPointer(dst))
	{
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dst), sizeof(T) * N));
		releaseUnpackedDeviceMemory = true;
	}
	else
	{
		d_dst = dst;
	}

	int* d_b = 0;  // device-bitmap

	if (src->HasBitmap())
	{
		CUDA_CHECK(cudaMalloc((void**)(&d_b), src->bitmapLength * sizeof(int)));
		CUDA_CHECK(cudaMemcpyAsync(d_b, src->bitmap, src->bitmapLength * sizeof(int), cudaMemcpyHostToDevice, *stream));
		CUDA_CHECK(cudaStreamSynchronize(*stream));
	}

	UnpackKernel<T><<<gridSize, blockSize, 0, *stream>>>(d_p, d_dst, d_b, src->coefficients, src->HasBitmap(), N);

	if (releaseUnpackedDeviceMemory)
	{
		CUDA_CHECK(cudaMemcpyAsync(dst, d_dst, sizeof(T) * N, cudaMemcpyDeviceToHost, *stream));
		CUDA_CHECK(cudaStreamSynchronize(*stream));
		CUDA_CHECK(cudaFree(d_dst));
	}

	CUDA_CHECK(cudaStreamSynchronize(*stream));
	CUDA_CHECK(cudaFree(d_p));

	if (d_b)
	{
		CUDA_CHECK(cudaFree(d_b));
	}

	if (destroyStream)
	{
		CUDA_CHECK(cudaStreamDestroy(*stream));
		delete stream;
	}
}

template __host__ void packing::Unpack<double>(const simple_packed* src, double* arr, cudaStream_t* stream);
template __host__ void packing::Unpack<float>(const simple_packed* src, float* arr, cudaStream_t* stream);
