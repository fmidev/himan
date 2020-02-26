namespace himan
{
namespace numerical_functions
{
template <typename T, class F, class G>
himan::matrix<T> Reduce2D(const himan::matrix<T>& A, const himan::matrix<T>& B, F&& f, G&& g, T init1, T init2)
{
	// find center position of kernel (half of kernel size)
	himan::matrix<T> ret(A.SizeX(), A.SizeY(), 1, A.MissingValue());

	T convolution_value;  // accumulated value of the convolution at a given grid point in A
	T kernel_weight_sum;  // accumulated value of the kernel weights in B that are used to compute the convolution
	                      // at given point A
	int ASizeX = int(A.SizeX());
	int ASizeY = int(A.SizeY());
	int BSizeX = int(B.SizeX());
	int BSizeY = int(B.SizeY());

	int kCenterX = BSizeX / 2;
	int kCenterY = BSizeY / 2;

	for (int j = kCenterY; j < ASizeY - kCenterY; ++j)  // columns
	{
		for (int i = kCenterX; i < ASizeX - kCenterX; ++i)  // rows
		{
			convolution_value = init1;
			kernel_weight_sum = init2;

			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;          // column index of flipped kernel
				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary
					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					const T a = A.At(ii, jj, 0);
					const T b = B.At(mm, nn, 0);
					f(convolution_value, kernel_weight_sum, a, b);
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = g(convolution_value, kernel_weight_sum);
		}
	}
	// treat boundaries separately
	// weights get adjusted so that the sum of weights for the active part of the kernel remains 1
	// calculate for upper boundary
	for (int j = 0; j < kCenterY; ++j)  // columns
	{
		for (int i = 0; i < ASizeX; ++i)  // rows
		{
			convolution_value = init1;
			kernel_weight_sum = init2;

			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;          // column index of flipped kernel
				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary

					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					// ignore input samples which are out of bound
					if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
					{
						const T a = A.At(ii, jj, 0);
						const T b = B.At(mm, nn, 0);
						f(convolution_value, kernel_weight_sum, a, b);
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = g(convolution_value, kernel_weight_sum);  // convolution_value / kernel_weight_sum;
		}
	}
	// calculate for lower boundary
	for (int j = ASizeY - kCenterY; j < ASizeY; ++j)  // columns
	{
		for (int i = 0; i < ASizeX; ++i)  // rows
		{
			convolution_value = init1;
			kernel_weight_sum = init2;

			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;  // column index of flipped kernel

				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary
					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					// ignore input samples which are out of bound
					if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
					{
						const T a = A.At(ii, jj, 0);
						const T b = B.At(mm, nn, 0);
						f(convolution_value, kernel_weight_sum, a, b);
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = g(convolution_value, kernel_weight_sum);  // convolution_value / kernel_weight_sum;
		}
	}
	// calculate for left boundary
	for (int j = 0; j < ASizeY; ++j)  // columns
	{
		for (int i = 0; i < kCenterX; ++i)  // rows
		{
			convolution_value = init1;
			kernel_weight_sum = init2;

			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;  // column index of flipped kernel

				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary
					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					// ignore input samples which are out of bound
					if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
					{
						const T a = A.At(ii, jj, 0);
						const T b = B.At(mm, nn, 0);
						f(convolution_value, kernel_weight_sum, a, b);
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = g(convolution_value, kernel_weight_sum);  // convolution_value / kernel_weight_sum;
		}
	}
	// calculate for right boundary
	for (int j = 0; j < ASizeY; ++j)  // columns
	{
		for (int i = ASizeX - kCenterX; i < ASizeX; ++i)  // rows
		{
			convolution_value = init1;
			kernel_weight_sum = init2;

			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;          // column index of flipped kernel
				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary
					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					// ignore input samples which are out of bound
					if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
					{
						const T a = A.At(ii, jj, 0);
						const T b = B.At(mm, nn, 0);
						f(convolution_value, kernel_weight_sum, a, b);
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = g(convolution_value, kernel_weight_sum);  // convolution_value / kernel_weight_sum;
		}
	}

	return ret;
}

template <typename T, class F>
himan::matrix<T> Prob2D(const himan::matrix<T>& A, const himan::matrix<T>& B, F&& f)
{
	return Reduce2D(A, B,
	                [=](T& val1, T& val2, const T& a, const T& b) {
		                if (IsValid(a * b))
		                {
			                val1 += f(a) ? b : T(0);
			                val2 += b;
		                }
	                },
	                [=](const T& val1, const T& val2) { return val2 == T(0) ? MissingValue<T>() : val1 / val2; }, T(0),
	                T(0));
}

template <typename T, class F>
himan::matrix<size_t> FindIndex2D(const himan::matrix<T>& A, const himan::matrix<T>& B, F&& f, T init1)
{
	// find center position of kernel (half of kernel size)
	himan::matrix<size_t> ret(A.SizeX(), A.SizeY(), 1, kHPMissingInt);

	T current_value;       //
	size_t current_index;  //

	int ASizeX = int(A.SizeX());
	int ASizeY = int(A.SizeY());
	int BSizeX = int(B.SizeX());
	int BSizeY = int(B.SizeY());

	int kCenterX = BSizeX / 2;
	int kCenterY = BSizeY / 2;

	for (int j = kCenterY; j < ASizeY - kCenterY; ++j)  // columns
	{
		for (int i = kCenterX; i < ASizeX - kCenterX; ++i)  // rows
		{
			current_value = init1;
			current_index = 0;

			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;          // column index of flipped kernel
				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary
					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					const T a = A.At(ii, jj, 0);
					const T b = B.At(mm, nn, 0);
					if (f(current_value, a, b))
					{
						current_index = A.Index(ii, jj, 0);
						current_value = a;
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = current_index;
		}
	}
	// treat boundaries separately
	// weights get adjusted so that the sum of weights for the active part of the kernel remains 1
	// calculate for upper boundary
	for (int j = 0; j < kCenterY; ++j)  // columns
	{
		for (int i = 0; i < ASizeX; ++i)  // rows
		{
			current_value = init1;
			current_index = 0;

			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;          // column index of flipped kernel
				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary

					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					// ignore input samples which are out of bound
					if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
					{
						const T a = A.At(ii, jj, 0);
						const T b = B.At(mm, nn, 0);
						if (f(current_value, a, b))
						{
							current_index = A.Index(ii, jj, 0);
							current_value = a;
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = current_index;
		}
	}
	// calculate for lower boundary
	for (int j = ASizeY - kCenterY; j < ASizeY; ++j)  // columns
	{
		for (int i = 0; i < ASizeX; ++i)  // rows
		{
			current_value = init1;
			current_index = 0;

			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;  // column index of flipped kernel

				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary
					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					// ignore input samples which are out of bound
					if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
					{
						const T a = A.At(ii, jj, 0);
						const T b = B.At(mm, nn, 0);
						if (f(current_value, a, b))
						{
							current_index = A.Index(ii, jj, 0);
							current_value = a;
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = current_index;
		}
	}
	// calculate for left boundary
	for (int j = 0; j < ASizeY; ++j)  // columns
	{
		for (int i = 0; i < kCenterX; ++i)  // rows
		{
			current_value = init1;
			current_index = 0;

			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;  // column index of flipped kernel

				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary
					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					// ignore input samples which are out of bound
					if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
					{
						const T a = A.At(ii, jj, 0);
						const T b = B.At(mm, nn, 0);
						if (f(current_value, a, b))
						{
							current_index = A.Index(ii, jj, 0);
							current_value = a;
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = current_index;
		}
	}
	// calculate for right boundary
	for (int j = 0; j < ASizeY; ++j)  // columns
	{
		for (int i = ASizeX - kCenterX; i < ASizeX; ++i)  // rows
		{
			current_value = init1;
			current_index = 0;

			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;          // column index of flipped kernel
				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary
					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					// ignore input samples which are out of bound
					if (ii >= 0 && ii < ASizeX && jj >= 0 && jj < ASizeY)
					{
						const T a = A.At(ii, jj, 0);
						const T b = B.At(mm, nn, 0);
						if (f(current_value, a, b))
						{
							current_index = A.Index(ii, jj, 0);
							current_value = a;
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = current_index;
		}
	}

	return ret;
}

#ifdef __CUDACC__
/**
 * @brief himan::matrix indexing for identical behaviour with the CPU Filter2D
 * @param W width of the matrix
 * @param H height of the matrix
 */
CUDA_DEVICE CUDA_INLINE size_t CudaMatrixIndex(size_t x, size_t y, size_t z, size_t W, size_t H)
{
	return z * W * H + y * W + x;
}

template <typename T, class F, class G>
CUDA_KERNEL void Reduce2DGPUKernel(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C, F f, G g,
                                   T init1, T init2, filter_opts opts)
{
	const int kCenterX = opts.bDimX / 2;
	const int kCenterY = opts.bDimY / 2;

	const int M = opts.aDimX;
	const int N = opts.aDimY;

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < M && j < N)
	{
		T convolutionValue = init1;
		T kernelWeightSum = init2;

		// kernel columns
		for (int n = 0; n < opts.bDimY; n++)
		{
			const int nn = opts.bDimY - 1 - n;

			// kernel rows
			for (int m = 0; m < opts.bDimX; m++)
			{
				const int mm = opts.bDimX - 1 - m;

				const int ii = i + (m - kCenterX);
				const int jj = j + (n - kCenterY);

				if (ii >= 0 && ii < M && jj >= 0 && jj < N)
				{
					const int aIdx = CudaMatrixIndex(ii, jj, 0, M, N);
					const int bIdx = CudaMatrixIndex(mm, nn, 0, opts.bDimX, opts.bDimY);
					const T aVal = A[aIdx];
					const T bVal = B[bIdx];

					f(convolutionValue, kernelWeightSum, aVal, bVal);
				}
			}
		}
		C[CudaMatrixIndex(i, j, 0, M, N)] = g(convolutionValue, kernelWeightSum);
	}
}

template <typename T, class F, class G>
himan::matrix<T> Reduce2DGPU(const himan::matrix<T>& A, const himan::matrix<T>& B, F&& f, G&& g, T init1, T init2)
{
	himan::matrix<T> C(A.SizeX(), A.SizeY(), 1, himan::MissingValue<T>());

	T* d_A = 0;
	T* d_B = 0;
	T* d_C = 0;

	CUDA_CHECK(cudaMalloc(&d_A, A.Size() * sizeof(T)));
	CUDA_CHECK(cudaMalloc(&d_B, B.Size() * sizeof(T)));
	CUDA_CHECK(cudaMalloc(&d_C, C.Size() * sizeof(T)));

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	CUDA_CHECK(cudaMemcpyAsync(d_A, A.ValuesAsPOD(), A.Size() * sizeof(T), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_B, B.ValuesAsPOD(), B.Size() * sizeof(T), cudaMemcpyHostToDevice, stream));

	numerical_functions::filter_opts opts = {static_cast<int>(A.SizeX()), static_cast<int>(A.SizeY()),
	                                         static_cast<int>(B.SizeX()), static_cast<int>(B.SizeY())};

	const int blockSizeX = 32;
	const int blockSizeY = 32;

	const auto gridSizeX = static_cast<int>(opts.aDimX / blockSizeX + (opts.aDimX % blockSizeX == 0 ? 0 : 1));
	const auto gridSizeY = opts.aDimY / blockSizeY + (opts.aDimY % blockSizeY == 0 ? 0 : 1);

	const dim3 gridSize(gridSizeX, gridSizeY);
	const dim3 blockSize(blockSizeX, blockSizeY);

	Reduce2DGPUKernel<<<gridSize, blockSize, 0, stream>>>(d_A, d_B, d_C, f, g, init1, init2, opts);

	CUDA_CHECK(cudaMemcpyAsync(C.ValuesAsPOD(), d_C, C.Size() * sizeof(T), cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaStreamSynchronize(stream));
	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_B));
	CUDA_CHECK(cudaFree(d_C));

	CUDA_CHECK(cudaStreamDestroy(stream));
	return C;
}

template <typename T, class F>
himan::matrix<T> Prob2DGPU(const himan::matrix<T>& A, const himan::matrix<T>& B, F&& f)
{
	return Reduce2DGPU(
	    A, B,
	    [=] __device__(T & val1, T & val2, const T& a, const T& b) {
		    if (IsValid(a * b))
		    {
			    val1 += f(a) ? b : T(0);
			    val2 += b;
		    }
	    },
	    [=] __device__(const T& val1, const T& val2) { return val2 == T(0) ? MissingValue<T>() : val1 / val2; }, T(0),
	    T(0));
}

#endif

}  // namespace numerical_functions
}  // namespace himan
