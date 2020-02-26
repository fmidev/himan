#include "numerical_functions.h"
#include "timer.h"
#include <algorithm>

using namespace himan;
using namespace numerical_functions;

#ifdef HAVE_CUDA

// GPU functions have definitions in this file because they have to compiled with nvcc
// and the direct header implementation cannot be visible to gcc

template <typename T>
matrix<T> numerical_functions::Filter2DGPU(const matrix<T>& A, const matrix<T>& B)
{
	return Reduce2DGPU(
	    A, B,
	    [=] __device__(T & val1, T & val2, const T& a, const T& b) {
		    if (IsValid(a * b))
		    {
			    val1 += a * b;
			    val2 += b;
		    }
	    },
	    [=] __device__(const T& val1, const T& val2) { return val2 == T(0) ? MissingValue<T>() : val1 / val2; }, T(0),
	    T(0));
}

template matrix<double> numerical_functions::Filter2DGPU(const matrix<double>&, const matrix<double>&);
template matrix<float> numerical_functions::Filter2DGPU(const matrix<float>&, const matrix<float>&);

template <typename T>
matrix<T> numerical_functions::Max2DGPU(const matrix<T>& A, const matrix<T>& B)
{
	return Reduce2DGPU(A, B,
	                   [] __device__(T & val1, T & val2, const T& a, const T& b) {
		                   if (IsValid(a * b))
			                   val1 = !(a * b <= val1) ? a : val1;
	                   },
	                   [] __device__(const T& val1, const T& val2) { return val1; }, MissingValue<T>(), T(0));
}

template matrix<double> numerical_functions::Max2DGPU(const matrix<double>&, const matrix<double>&);
template matrix<float> numerical_functions::Max2DGPU(const matrix<float>&, const matrix<float>&);

template <typename T>
matrix<T> numerical_functions::Min2DGPU(const matrix<T>& A, const matrix<T>& B)
{
	return Reduce2DGPU(A, B,
	                   [] __device__(T & val1, T & val2, const T& a, const T& b) {
		                   if (IsValid(a * b))
			                   val1 = !(a * b >= val1) ? a : val1;
	                   },
	                   [] __device__(const T& val1, const T& val2) { return val1; }, MissingValue<T>(), T(0));
}

template matrix<double> numerical_functions::Min2DGPU(const matrix<double>&, const matrix<double>&);
template matrix<float> numerical_functions::Min2DGPU(const matrix<float>&, const matrix<float>&);

template <typename T>
matrix<T> numerical_functions::ProbLimitGt2DGPU(const matrix<T>& A, const matrix<T>& B, T limit)
{
	return Prob2DGPU<T>(A, B, [=] __device__(const T& val) { return val > limit; });
}

template matrix<double> numerical_functions::ProbLimitGt2DGPU(const matrix<double>&, const matrix<double>&, double);
template matrix<float> numerical_functions::ProbLimitGt2DGPU(const matrix<float>&, const matrix<float>&, float);

template <typename T>
matrix<T> numerical_functions::ProbLimitGe2DGPU(const matrix<T>& A, const matrix<T>& B, T limit)
{
	return Prob2DGPU<T>(A, B, [=] __device__(const T& val) { return val >= limit; });
}

template matrix<double> numerical_functions::ProbLimitGe2DGPU(const matrix<double>&, const matrix<double>&, double);
template matrix<float> numerical_functions::ProbLimitGe2DGPU(const matrix<float>&, const matrix<float>&, float);

template <typename T>
matrix<T> numerical_functions::ProbLimitLt2DGPU(const matrix<T>& A, const matrix<T>& B, T limit)
{
	return Prob2DGPU<T>(A, B, [=] __device__(const T& val) { return val < limit; });
}

template matrix<double> numerical_functions::ProbLimitLt2DGPU(const matrix<double>&, const matrix<double>&, double);
template matrix<float> numerical_functions::ProbLimitLt2DGPU(const matrix<float>&, const matrix<float>&, float);

template <typename T>
matrix<T> numerical_functions::ProbLimitLe2DGPU(const matrix<T>& A, const matrix<T>& B, T limit)
{
	return Prob2DGPU<T>(A, B, [=] __device__(const T& val) { return val <= limit; });
}

template matrix<double> numerical_functions::ProbLimitLe2DGPU(const matrix<double>&, const matrix<double>&, double);
template matrix<float> numerical_functions::ProbLimitLe2DGPU(const matrix<float>&, const matrix<float>&, float);

template <typename T>
matrix<T> numerical_functions::ProbLimitEq2DGPU(const matrix<T>& A, const matrix<T>& B, T limit)
{
	return Prob2DGPU<T>(A, B, [=] __device__(const T& val) { return val == limit; });
}

template matrix<double> numerical_functions::ProbLimitEq2DGPU(const matrix<double>&, const matrix<double>&, double);
template matrix<float> numerical_functions::ProbLimitEq2DGPU(const matrix<float>&, const matrix<float>&, float);

#endif
