namespace himan
{
namespace numerical_functions
{
template <class T, class S>
himan::matrix<double> Reduce2D(const himan::matrix<double>& A, const himan::matrix<double>& B, T&& f, S&& g,
                               double init1, double init2)
{
	// find center position of kernel (half of kernel size)
	himan::matrix<double> ret(A.SizeX(), A.SizeY(), 1, A.MissingValue());

	double convolution_value;  // accumulated value of the convolution at a given grid point in A
	double kernel_weight_sum;  // accumulated value of the kernel weights in B that are used to compute the convolution
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

					const double a = A.At(ii, jj, 0);
					const double b = B.At(mm, nn, 0);
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
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);
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
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);
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
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);
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
						const double a = A.At(ii, jj, 0);
						const double b = B.At(mm, nn, 0);
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

template <class T>
himan::matrix<double> Prob2D(const himan::matrix<double>& A, const himan::matrix<double>& B, T&& f)
{
	return Reduce2D(A, B,
	                [=](double& val1, double& val2, const double& a, const double& b) {
		                if (IsValid(a * b))
		                {
			                val1 += f(a) ? b : 0.0;
			                val2 += b;
		                }
	                },
	                [](const double& val1, const double& val2) { return val2 == 0.0 ? MissingDouble() : val1 / val2; },
	                0.0, 0.0);
}

}  // namespace numerical_functions
}  // namespace himan
