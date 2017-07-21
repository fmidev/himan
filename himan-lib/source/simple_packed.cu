/**
 * @file simple_packed.cu
 *
 */

#include "simple_packed.h"

#include <NFmiGribPacking.h>
#include <cub/cub.cuh>
#include <grib_api.h>

#include "cuda_helper.h"

using namespace himan;

long get_binary_scale_fact(double max, double min, long bpval)
{
	assert(max >= min);
	double range = max - min;
	double zs = 1;
	long scale = 0;
	const long last = 127; /* Depends on edition, should be parameter */

	unsigned long maxint = packed_data_util::GetGribPower(bpval, 2) - 1;
	double dmaxint = (double)maxint;

	assert(bpval >= 1);

	if (range == 0) return 0;

	/* range -= 1e-10; */
	while ((range * zs) <= dmaxint)
	{
		scale--;
		zs *= 2;
	}

	while ((range * zs) > dmaxint)
	{
		scale++;
		zs /= 2;
	}

	while ((unsigned long)(range * zs + 0.5) <= maxint)
	{
		scale--;
		zs *= 2;
	}

	while ((unsigned long)(range * zs + 0.5) > maxint)
	{
		scale++;
		zs /= 2;
	}

	if (scale < -last)
	{
		printf("grib_get_binary_scale_fact: max=%g min=%g\n", max, min);
		scale = -last;
	}
	assert(scale <= last);

	return scale;
}

long get_decimal_scale_fact(double max, double min, long bpval, long binary_scale)
{
	assert(max >= min);

	double range = max - min;
	const long last = 127; /* Depends on edition, should be parameter */
	double decimal_scale_factor = 0;
	double f;
	double minrange = 0, maxrange = 0;
	double decimal = 1;
	long bits_per_value = bpval;

	double unscaled_min = min;
	double unscaled_max = max;

	f = packed_data_util::GetGribPower(bits_per_value, 2) - 1;
	minrange = packed_data_util::GetGribPower(-last, 2) * f;
	maxrange = packed_data_util::GetGribPower(last, 2) * f;

	while (range < minrange)
	{
		decimal_scale_factor += 1;
		decimal *= 10;
		min = unscaled_min * decimal;
		max = unscaled_max * decimal;
		range = (max - min);
	}
	while (range > maxrange)
	{
		decimal_scale_factor -= 1;
		decimal /= 10;
		min = unscaled_min * decimal;
		max = unscaled_max * decimal;
		range = (max - min);
	}

	return decimal_scale_factor;
}

template <typename T>
__host__ T simple_packed::Min(T* d_arr, size_t N, cudaStream_t& stream)
{
	void* d_temp = 0;
	size_t temp_N = 0;
	T* d_min = 0;
	T min;

	CUDA_CHECK(cudaMalloc((void**)&d_min, sizeof(T)));

	// Allocate temp storage
	CUDA_CHECK(cub::DeviceReduce::Min(d_temp, temp_N, d_arr, d_min, N, stream));
	CUDA_CHECK(cudaMalloc((void**)&d_temp, temp_N));

	CUDA_CHECK(cub::DeviceReduce::Min(d_temp, temp_N, d_arr, d_min, N, stream));

	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaFree(d_temp));
	CUDA_CHECK(cudaMemcpyAsync(&min, d_min, sizeof(T), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaFree(d_min));

	return min;
}

template <typename T>
__host__ T simple_packed::Max(T* d_arr, size_t N, cudaStream_t& stream)
{
	void* d_temp = 0;
	size_t temp_N = 0;
	T* d_max = 0;
	T max;

	CUDA_CHECK(cudaMalloc((void**)&d_max, sizeof(T)));

	// Allocate temp storage
	CUDA_CHECK(cub::DeviceReduce::Max(d_temp, temp_N, d_arr, d_max, N, stream));
	CUDA_CHECK(cudaMalloc((void**)&d_temp, temp_N));

	CUDA_CHECK(cub::DeviceReduce::Max(d_temp, temp_N, d_arr, d_max, N, stream));

	CUDA_CHECK(cudaStreamSynchronize(stream));

	CUDA_CHECK(cudaFree(d_temp));
	CUDA_CHECK(cudaMemcpyAsync(&max, d_max, sizeof(T), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaFree(d_max));

	return max;
}

__host__ void simple_packed::Unpack(double* arr, size_t N, cudaStream_t* stream)
{
	assert(arr);
	assert(N > 0);

	if (packedLength == 0 && coefficients.bitsPerValue == 0)
	{
		// Special case for static grid

		// For empty grid (all values missing), grib_api gives reference value 1!

		double fillValue = coefficients.referenceValue;

		if (HasBitmap())
		{
			// Make an assumption: if grid is static and bitmap is defined, it is probably
			// all missing.

			fillValue = himan::MissingDouble();
		}

		if (NFmiGribPacking::IsHostPointer(arr))
		{
			std::fill(arr, arr + N, fillValue);
		}
		else
		{
			NFmiGribPacking::Fill(arr, N, fillValue);
		}

		return;
	}

	if (N != unpackedLength)
	{
		std::cerr << "Error::" << ClassName() << " Allocated memory size is different from data: " << N << " vs "
		          << unpackedLength << std::endl;
		return;
	}

	// We need to create a stream if no stream is specified since dereferencing
	// a null pointer is, well, not a good thing.

	bool destroyStream = false;

	if (!stream)
	{
		stream = new cudaStream_t;
		CUDA_CHECK(cudaStreamCreate(stream));
		destroyStream = true;
	}

	// Allocate memory on device for packed data

	unsigned char* d_p = 0;  // device-packed data

	CUDA_CHECK(cudaMalloc((void**)(&d_p), packedLength * sizeof(unsigned char)));
	CUDA_CHECK(cudaMemcpyAsync(d_p, data, packedLength * sizeof(unsigned char), cudaMemcpyHostToDevice, *stream));

	// Allocate memory on device for unpacked data

	double* d_arr = 0;

	bool releaseUnpackedDeviceMemory = false;

	if (NFmiGribPacking::IsHostPointer(arr))
	{
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_arr), sizeof(double) * N));
		releaseUnpackedDeviceMemory = true;
	}
	else
	{
		d_arr = arr;
	}

	int blockSize = 512;
	int gridSize = unpackedLength / blockSize + (unpackedLength % blockSize == 0 ? 0 : 1);

	int* d_b = 0;  // device-bitmap

	if (HasBitmap())
	{
		CUDA_CHECK(cudaMalloc((void**)(&d_b), bitmapLength * sizeof(int)));
		CUDA_CHECK(cudaMemcpyAsync(d_b, bitmap, bitmapLength * sizeof(int), cudaMemcpyHostToDevice, *stream));
		CUDA_CHECK(cudaStreamSynchronize(*stream));
	}

	simple_packed_util::Unpack<<<gridSize, blockSize, 0, *stream>>>(d_p, d_arr, d_b, coefficients, HasBitmap(),
	                                                                unpackedLength);

	if (releaseUnpackedDeviceMemory)
	{
		CUDA_CHECK(cudaMemcpyAsync(arr, d_arr, sizeof(double) * N, cudaMemcpyDeviceToHost, *stream));
		CUDA_CHECK(cudaStreamSynchronize(*stream));
		CUDA_CHECK(cudaFree(d_arr));
	}

	CUDA_CHECK(cudaStreamSynchronize(*stream));
	CUDA_CHECK(cudaFree(d_p));

	if (HasBitmap())
	{
		CUDA_CHECK(cudaFree(d_b));
	}

	if (destroyStream)
	{
		CUDA_CHECK(cudaStreamDestroy(*stream));
		delete stream;
	}
}

__global__ void print(double* d)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	printf("%d %f\n", idx, d[idx]);
}

CUDA_HOST
void simple_packed::Pack(double* d_arr, size_t N, cudaStream_t* stream)
{
#ifdef GRIB_WRITE_PACKED_DATA

	if (packedLength)
	{
		std::cerr << "Data already packed" << std::endl;
		return;
	}

	if (!d_arr)
	{
		std::cerr << "Memory not allocated for unpacked data" << std::endl;
		return;
	}

	// We need to create a stream if no stream is specified since dereferencing
	// a null pointer is, well, not a good thing.

	bool destroyStream = false;

	if (!stream)
	{
		stream = new cudaStream_t;
		CUDA_CHECK(cudaStreamCreate(stream));
		destroyStream = true;
	}

	// Allocate memory on host

	if (!data)
	{
		packedLength = ((coefficients.bitsPerValue * N) + 7) / 8;
		unpackedLength = N;

		CUDA_CHECK(cudaMallocHost((void**)&data, packedLength * sizeof(unsigned char)));
	}
	else
	{
		std::cerr << "Data not packed but memory already allocated?" << std::endl;
		return;
	}

	// Allocate memory on device for packed data and transfer data to device

	unsigned char* d_p = 0;  // device-packed data
	CUDA_CHECK(cudaMalloc((void**)(&d_p), packedLength * sizeof(unsigned char)));

	/*
	 * 1. Get unpacked data range
	 * 2. Calculate coefficients
	 * 3. Reduce
	 */

	// 1. Get unpacked data range

	assert(d_arr);

	double max = Max(d_arr, N, *stream);
	double min = Min(d_arr, N, *stream);

	assert(isfinite(max) && isfinite(min));

#ifdef DEBUG
	std::cout << "min: " << min << " max: " << max << std::endl;
#endif

	int blockSize = 512;
	int gridSize = unpackedLength / blockSize + (unpackedLength % blockSize == 0 ? 0 : 1);

	// 2. Calculate coefficients

	coefficients.binaryScaleFactor = get_binary_scale_fact(max, min, coefficients.bitsPerValue);
	coefficients.decimalScaleFactor =
	    get_decimal_scale_fact(max, min, coefficients.bitsPerValue, coefficients.binaryScaleFactor);

	grib_handle* h = grib_handle_new_from_samples(0, "GRIB1");
	GRIB_CHECK(grib_get_reference_value(h, min, &coefficients.referenceValue), 0);
	GRIB_CHECK(grib_handle_delete(h), 0);

	if (HasBitmap())
	{
		std::cerr << "bitmap packing not supported yet" << std::endl;
		abort();
	}

	// 3. Reduce

	simple_packed_util::Pack<<<gridSize, blockSize, 0, *stream>>>(d_p, d_arr, 0, coefficients, 0, unpackedLength);

	CUDA_CHECK(cudaStreamSynchronize(*stream));
	CUDA_CHECK_ERROR_MSG("Kernel invocation");

	CUDA_CHECK(cudaMemcpyAsync(data, d_p, sizeof(unsigned char) * packedLength, cudaMemcpyDeviceToHost, *stream));
	CUDA_CHECK(cudaStreamSynchronize(*stream));

	CUDA_CHECK(cudaFree(d_p));

	if (destroyStream)
	{
		CUDA_CHECK(cudaStreamDestroy(*stream));
		delete stream;
	}
#else
	std::cerr << "PACKING NOT SUPPORTED" << std::endl;
#endif
}

__global__ void simple_packed_util::Unpack(unsigned char* d_p, double* d_u, int* d_b, simple_packed_coefficients coeff,
                                           bool hasBitmap, size_t N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		if (coeff.bitsPerValue %
		    8)  // modulo is expensive but "Compiler will convert literal power-of-2 divides to bitwise shifts"
		{
			UnpackUnevenBytes(d_p, d_u, d_b, coeff, hasBitmap, idx);
		}
		else
		{
			UnpackFullBytes(d_p, d_u, d_b, coeff, hasBitmap, idx);
		}
	}
}
/*
__host__ __device__
double simple_packed_util::GetGribPower(long s,long n)
{
    double divisor = 1.0;
    while(s < 0)
    {
        divisor /= n;
        s++;
    }
    while(s > 0)
    {
        divisor *= n;
        s--;
    }
    return divisor;
}
*/
__device__ void simple_packed_util::GetBitValue(unsigned char* p, long bitp, int* val)
{
	p += (bitp >> 3);
	*val = (*p & (1 << (7 - (bitp % 8))));
}

__device__ void simple_packed_util::UnpackFullBytes(unsigned char* __restrict__ d_p, double* __restrict__ d_u,
                                                    int* __restrict__ d_b, simple_packed_coefficients coeff,
                                                    bool hasBitmap, int idx)
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

__device__ void simple_packed_util::UnpackUnevenBytes(unsigned char* __restrict__ d_p, double* __restrict__ d_u,
                                                      int* __restrict__ d_b, simple_packed_coefficients coeff,
                                                      bool hasBitmap, int idx)
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

			if (val) lvalue += 1;

			bitp += 1;
		}

		d_u[idx] = fma(lvalue, coeff.binaryScaleFactor, coeff.referenceValue) * coeff.decimalScaleFactor;
	}
}

__device__ void simple_packed_util::PackUnevenBytes(unsigned char* __restrict__ d_p, const double* __restrict__ d_u,
                                                    size_t values_len, simple_packed_coefficients coeff, int idx)
{
	double decimal = packed_data_util::GetGribPower(-coeff.decimalScaleFactor, 10);
	double divisor = packed_data_util::GetGribPower(-coeff.binaryScaleFactor, 2);

	double x = (((d_u[idx] * decimal) - coeff.referenceValue) * divisor) + 0.5;
	unsigned long unsigned_val = static_cast<unsigned long>(x);

	long bitp = coeff.bitsPerValue * idx;

	long i = 0;

	for (i = coeff.bitsPerValue - 1; i >= 0; i--)
	{
		if (BitTest(unsigned_val, i))
		{
			SetBitOn(d_p, bitp);
		}
		else
		{
			SetBitOff(d_p, bitp);
		}

		bitp++;
	}
}

__device__ void simple_packed_util::PackFullBytes(unsigned char* __restrict__ d_p, const double* __restrict__ d_u,
                                                  size_t values_len, simple_packed_coefficients coeff, int idx)
{
	double decimal = packed_data_util::GetGribPower(-coeff.decimalScaleFactor, 10);
	double divisor = packed_data_util::GetGribPower(-coeff.binaryScaleFactor, 2);

	// unsigned char* encoded = d_p + idx * static_cast<int> (coefficients.bpv/8);

	double x = ((((d_u[idx] * decimal) - coeff.referenceValue) * divisor) + 0.5);
	unsigned long unsigned_val = static_cast<unsigned long>(x);

	unsigned char* encoded = &d_p[idx * static_cast<int>(coeff.bitsPerValue / 8)];

	while (coeff.bitsPerValue >= 8)
	{
		coeff.bitsPerValue -= 8;
		*encoded = (unsigned_val >> coeff.bitsPerValue);
		encoded++;
	}
}

__global__ void simple_packed_util::Pack(unsigned char* d_p, double* d_u, int* d_b, simple_packed_coefficients coeff,
                                         bool hasBitmap, size_t N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		if (coeff.bitsPerValue %
		    8)  // modulo is expensive but "Compiler will convert literal power-of-2 divides to bitwise shifts"
		{
			PackUnevenBytes(d_p, d_u, N, coeff, idx);
		}
		else
		{
			PackFullBytes(d_p, d_u, N, coeff, idx);
		}
	}
}

__device__ void simple_packed_util::SetBitOn(unsigned char* p, long bitp)
{
	p += bitp / 8;
	*p |= (1u << (7 - ((bitp) % 8)));
}

__device__ void simple_packed_util::SetBitOff(unsigned char* p, long bitp)
{
	p += bitp / 8;
	*p &= ~(1u << (7 - ((bitp) % 8)));
}
