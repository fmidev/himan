/**
 * @file numerical_functions.cpp
 */

#include "numerical_functions.h"
#include "NFmiInterpolation.h"
#include "plugin_factory.h"
#include <algorithm>

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"
#include "radon.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan;
using namespace numerical_functions;

matrix<double> numerical_functions::Filter2D(const matrix<double>& A, const matrix<double>& B)
{
	// find center position of kernel (half of kernel size)
	matrix<double> ret(A.SizeX(), A.SizeY(), 1, A.MissingValue());

	double convolution_value;  // accumulated value of the convolution at a given grid point in A
	double kernel_weight_sum;  // accumulated value of the kernel weights in B that are used to compute the convolution
	                           // at given point A

	int ASizeX = int(A.SizeX());
	int ASizeY = int(A.SizeY());
	int BSizeX = int(B.SizeX());
	int BSizeY = int(B.SizeY());

	int kCenterX = BSizeX / 2;
	int kCenterY = BSizeY / 2;

	// check if data contains missing values
	if (A.MissingCount() == 0)  // if no missing values in the data we can use a faster algorithm
	{
		// calculate for inner field
		// the weights are used as given on input
		// ASSERT (sum(B) == 1)
		for (int j = kCenterY; j < ASizeY - kCenterY; ++j)  // columns
		{
			for (int i = kCenterX; i < ASizeX - kCenterX; ++i)  // rows
			{
				convolution_value = 0;
				for (int n = 0; n < BSizeY; ++n)  // kernel columns
				{
					int nn = BSizeY - 1 - n;          // column index of flipped kernel
					for (int m = 0; m < BSizeX; ++m)  // kernel rows
					{
						int mm = BSizeX - 1 - m;  // row index of flipped kernel

						// index of input signal, used for checking boundary
						int ii = i + (m - kCenterX);
						int jj = j + (n - kCenterY);
						convolution_value += A.At(ii, jj, 0) * B.At(mm, nn, 0);
					}
				}
				const size_t index = ret.Index(i, j, 0);
				ret[index] = convolution_value;
			}
		}

		// treat boundaries separately
		// weights get adjusted so that the sum of weights for the active part of the kernel remains 1
		// calculate for upper boundary
		for (int j = 0; j < kCenterY; ++j)  // columns
		{
			for (int i = 0; i < ASizeX; ++i)  // rows
			{
				convolution_value = 0;
				kernel_weight_sum = 0;
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
							convolution_value += A.At(ii, jj, 0) * B.At(mm, nn, 0);
							kernel_weight_sum += B.At(mm, nn, 0);
						}
					}
				}
				const size_t index = ret.Index(i, j, 0);
				ret[index] = convolution_value / kernel_weight_sum;
			}
		}

		// calculate for lower boundary
		for (int j = ASizeY - kCenterY; j < ASizeY; ++j)  // columns
		{
			for (int i = 0; i < ASizeX; ++i)  // rows
			{
				convolution_value = 0;
				kernel_weight_sum = 0;
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
							convolution_value += A.At(ii, jj, 0) * B.At(mm, nn, 0);
							kernel_weight_sum += B.At(mm, nn, 0);
						}
					}
				}
				const size_t index = ret.Index(i, j, 0);
				ret[index] = convolution_value / kernel_weight_sum;
			}
		}

		// calculate for left boundary
		for (int j = 0; j < ASizeY; ++j)  // columns
		{
			for (int i = 0; i < kCenterX; ++i)  // rows
			{
				convolution_value = 0;
				kernel_weight_sum = 0;
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
							convolution_value += A.At(ii, jj, 0) * B.At(mm, nn, 0);
							kernel_weight_sum += B.At(mm, nn, 0);
						}
					}
				}
				const size_t index = ret.Index(i, j, 0);
				ret[index] = convolution_value / kernel_weight_sum;
			}
		}

		// calculate for right boundary
		for (int j = 0; j < ASizeY; ++j)  // columns
		{
			for (int i = ASizeX - kCenterX; i < ASizeX; ++i)  // rows
			{
				convolution_value = 0;
				kernel_weight_sum = 0;
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
							convolution_value += A.At(ii, jj, 0) * B.At(mm, nn, 0);
							kernel_weight_sum += B.At(mm, nn, 0);
						}
					}
				}
				const size_t index = ret.Index(i, j, 0);
				ret[index] = convolution_value / kernel_weight_sum;
			}
		}
	}
	else  // data contains missing values
	{
		std::cout << "util::Filter2D: Data contains missing values -> Choosing slow algorithm." << std::endl;
		double kernel_missing_count;
		for (int j = 0; j < ASizeY; ++j)  // columns
		{
			for (int i = 0; i < ASizeX; ++i)  // rows
			{
				convolution_value = 0;
				kernel_weight_sum = 0;
				kernel_missing_count = 0;
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
							if (A.IsMissing(ii, jj, 0))
							{
								kernel_missing_count++;
								continue;
							}

							convolution_value += A.At(ii, jj, 0) * B.At(mm, nn, 0);
							kernel_weight_sum += B.At(mm, nn, 0);
						}
					}
				}
				if (kernel_missing_count < 3)
				{
					const size_t index = ret.Index(i, j, 0);
					ret[index] = convolution_value / kernel_weight_sum;
				}
				else
				{
					const size_t index = ret.Index(i, j, 0);
					ret[index] = himan::MissingDouble();
				}
			}
		}
	}
	return ret;
}

himan::matrix<double> numerical_functions::Max2D(const himan::matrix<double>& A, const himan::matrix<double>& B)
{
	using himan::MissingDouble;

	// find center position of kernel (half of kernel size)
	himan::matrix<double> ret(A.SizeX(), A.SizeY(), 1, A.MissingValue());

	double max_value;  // maximum value of the convolution

	int ASizeX = int(A.SizeX());
	int ASizeY = int(A.SizeY());
	int BSizeX = int(B.SizeX());
	int BSizeY = int(B.SizeY());

	int kCenterX = BSizeX / 2;
	int kCenterY = BSizeY / 2;

	// calculate for inner field
	// the weights are used as given on input
	// ASSERT (sum(B) == 1)

	ASSERT(B.MissingCount() == 0);

	for (int j = kCenterY; j < ASizeY - kCenterY; ++j)  // columns
	{
		for (int i = kCenterX; i < ASizeX - kCenterX; ++i)  // rows
		{
			max_value = -1e38;
			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;          // column index of flipped kernel
				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary
					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					const double a = A.At(ii, jj, 0);
					const double b = B.At(mm, nn, 0);

					if (IsValid(a) && b != 0)
					{
						max_value = fmax(a * b, max_value);
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = (max_value == -1e38 ? MissingDouble() : max_value);
		}
	}

	// treat boundaries separately
	// calculate for upper boundary
	for (int j = 0; j < kCenterY; ++j)  // columns
	{
		for (int i = 0; i < ASizeX; ++i)  // rows
		{
			max_value = -1e38;

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
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);

						if (IsValid(a) && b != 0)
						{
							max_value = fmax(a * b, max_value);
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = (max_value == -1e38 ? MissingDouble() : max_value);
		}
	}

	// calculate for lower boundary
	for (int j = ASizeY - kCenterY; j < ASizeY; ++j)  // columns
	{
		for (int i = 0; i < ASizeX; ++i)  // rows
		{
			max_value = -1e38;

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
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);

						if (IsValid(a) && b != 0)
						{
							max_value = fmax(a * b, max_value);
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = (max_value == -1e38 ? MissingDouble() : max_value);
		}
	}

	// calculate for left boundary
	for (int j = 0; j < ASizeY; ++j)  // columns
	{
		for (int i = 0; i < kCenterX; ++i)  // rows
		{
			max_value = -1e38;

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
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);

						if (IsValid(a) && b != 0)
						{
							max_value = fmax(a * b, max_value);
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = (max_value == -1e38 ? MissingDouble() : max_value);
		}
	}

	// calculate for right boundary
	for (int j = 0; j < ASizeY; ++j)  // columns
	{
		for (int i = ASizeX - kCenterX; i < ASizeX; ++i)  // rows
		{
			max_value = -1e38;

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
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);

						if (IsValid(a) && b != 0)
						{
							max_value = fmax(a * b, max_value);
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = (max_value == -1e38 ? MissingDouble() : max_value);
		}
	}

	return ret;
}

himan::matrix<double> numerical_functions::Min2D(const himan::matrix<double>& A, const himan::matrix<double>& B)
{
	using himan::MissingDouble;

	// find center position of kernel (half of kernel size)
	himan::matrix<double> ret(A.SizeX(), A.SizeY(), 1, A.MissingValue());

	double min_value;  // minimum value of the convolution

	int ASizeX = int(A.SizeX());
	int ASizeY = int(A.SizeY());
	int BSizeX = int(B.SizeX());
	int BSizeY = int(B.SizeY());

	int kCenterX = BSizeX / 2;
	int kCenterY = BSizeY / 2;

	// calculate for inner field
	// the weights are used as given on input
	// ASSERT (sum(B) == 1)

	ASSERT(B.MissingCount() == 0);

	for (int j = kCenterY; j < ASizeY - kCenterY; ++j)  // columns
	{
		for (int i = kCenterX; i < ASizeX - kCenterX; ++i)  // rows
		{
			min_value = 1e38;
			for (int n = 0; n < BSizeY; ++n)  // kernel columns
			{
				int nn = BSizeY - 1 - n;          // column index of flipped kernel
				for (int m = 0; m < BSizeX; ++m)  // kernel rows
				{
					int mm = BSizeX - 1 - m;  // row index of flipped kernel

					// index of input signal, used for checking boundary
					int ii = i + (m - kCenterX);
					int jj = j + (n - kCenterY);

					const double a = A.At(ii, jj, 0);
					const double b = B.At(mm, nn, 0);

					if (IsValid(a) && b != 0)
					{
						min_value = fmin(a * b, min_value);
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = (min_value == 1e38 ? MissingDouble() : min_value);
		}
	}

	// treat boundaries separately
	// calculate for upper boundary
	for (int j = 0; j < kCenterY; ++j)  // columns
	{
		for (int i = 0; i < ASizeX; ++i)  // rows
		{
			min_value = 1e38;

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
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);

						if (IsValid(a) && b != 0)
						{
							min_value = fmin(a * b, min_value);
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = (min_value == 1e38 ? MissingDouble() : min_value);
		}
	}

	// calculate for lower boundary
	for (int j = ASizeY - kCenterY; j < ASizeY; ++j)  // columns
	{
		for (int i = 0; i < ASizeX; ++i)  // rows
		{
			min_value = 1e38;

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
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);

						if (IsValid(a) && b != 0)
						{
							min_value = fmin(a * b, min_value);
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = (min_value == 1e38 ? MissingDouble() : min_value);
		}
	}

	// calculate for left boundary
	for (int j = 0; j < ASizeY; ++j)  // columns
	{
		for (int i = 0; i < kCenterX; ++i)  // rows
		{
			min_value = 1e38;

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
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);

						if (IsValid(a) && b != 0)
						{
							min_value = fmin(a * b, min_value);
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = (min_value == 1e38 ? MissingDouble() : min_value);
		}
	}

	// calculate for right boundary
	for (int j = 0; j < ASizeY; ++j)  // columns
	{
		for (int i = ASizeX - kCenterX; i < ASizeX; ++i)  // rows
		{
			min_value = 1e38;

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
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);

						if (IsValid(a) && b != 0)
						{
							min_value = fmin(a * b, min_value);
						}
					}
				}
			}
			const size_t index = ret.Index(i, j, 0);
			ret[index] = (min_value == 1e38 ? MissingDouble() : min_value);
		}
	}

	return ret;
}
