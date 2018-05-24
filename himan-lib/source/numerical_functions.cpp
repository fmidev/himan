/**
 * @file numerical_functions.cpp
 */

#include "numerical_functions.h"
#include "numerical_functions_impl.h"

using namespace himan;
using namespace numerical_functions;

matrix<double> numerical_functions::Filter2D(const matrix<double>& A, const matrix<double>& B)
{
	return Reduce2D(A, B,
	                [](double& val1, double& val2, const double& a, const double& b) {
		                if (IsValid(a * b))
		                {
			                val1 += a * b;
			                val2 += b;
		                }
	                },
	                [](const double& val1, const double& val2) { return val2 == 0.0 ? MissingDouble() : val1 / val2; },
	                0.0, 0.0);
}

matrix<double> numerical_functions::Max2D(const matrix<double>& A, const matrix<double>& B)
{
	return Reduce2D(A, B,
	                [](double& val1, double& val2, const double& a, const double& b) {
		                if (IsValid(a * b))
			                val1 = !(a * b <= val1) ? a : val1;
	                },
	                [](const double& val1, const double& val2) { return val1; }, MissingDouble(), 0.0);
}

matrix<double> numerical_functions::Min2D(const matrix<double>& A, const matrix<double>& B)
{
	return Reduce2D(A, B,
	                [](double& val1, double& val2, const double& a, const double& b) {
		                if (IsValid(a * b))
			                val1 = !(a * b >= val1) ? a : val1;
	                },
	                [](const double& val1, const double& val2) { return val1; }, MissingDouble(), 0.0);
}
