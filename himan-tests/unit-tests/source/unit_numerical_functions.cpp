#include "himan_unit.h"
#include "numerical_functions.h"

#include "numerical_functions_helper.h"

#include "timer.h"

#define BOOST_TEST_MODULE numerical_functions

using namespace std;
using namespace himan;

const double kEpsilon = 1e-3;

void Dump(const himan::matrix<double>& m)
{
	for (size_t i=0; i < m.SizeX();++i){
    		for (size_t j=0; j < m.SizeY();++j){
      			std::cout << m.At(i,j,0) << " ";
    		}
    		std::cout << std::endl;
  	}

}

BOOST_AUTO_TEST_CASE(FILTER2D)
{
	// Filter a plane with given filter kernel
    
	// Declare matrices
	himan::matrix<double> A(11,8,1,kFloatMissing);
	himan::matrix<double> B(3,3,1,kFloatMissing);
	himan::matrix<double> C;
	himan::matrix<double> D(11,8,1,kFloatMissing);

        FilterTestSetup(A, B, D);

	// Compute smoothened matrix
	C = himan::numerical_functions::Filter2D(A,B);

	// Compare results
	for(size_t i=0; i < C.Size(); ++i)
	{
		BOOST_CHECK_CLOSE(C.At(i),D.At(i),kEpsilon);
	}

	// computed filtered matrix
	std::cout << "Matrix C computed with Filter2D:" << std::endl;
	Dump(C);

	std::cout << std::endl << "Matrix D as reference case for Filter2D computation:" << std::endl; 
	Dump(D);

}

BOOST_AUTO_TEST_CASE(FILTER2D_LARGE)
{
    // Filter a plane with given filter kernel
    
    // Declare matrices
    himan::matrix<double> A(2001,1000,1,kFloatMissing);
    himan::matrix<double> B(3,3,1,kFloatMissing);
    himan::matrix<double> D(2001,1000,1,kFloatMissing);

    FilterTestSetup(A, B, D);

    himan::timer timer;
    timer.Start();

    // Compute smoothened matrix
    himan::matrix<double> C = himan::numerical_functions::Filter2D(A,B);

    timer.Stop();

    // Compare results
    for(size_t i=0; i < C.Size(); ++i)
    {
        BOOST_CHECK_CLOSE(C.At(i),D.At(i),kEpsilon);
    }

    std::cout << "Filter2D took " << timer.GetTime() << " ms " << std::endl;

}

BOOST_AUTO_TEST_CASE(MAX2D)
{
	himan::matrix<double> A(5, 5, 1, kFloatMissing);
	A.Set({1, 2, 3, 4, 5,
		6, 7, 8, 9, 0,
		1, 2, 3, 4, 5,
		6, 7, 8, 9, 0,
		1, 2, 3, 4, 5});


	himan::matrix<double> B(3, 3, 1, kFloatMissing);
	B.Set({0, 1, 0, 1, 1, 1, 0, 1, 0});

	auto result = numerical_functions::Max2D(A, B);

	std::cout << "source" << std::endl;
	Dump(A);
	std::cout << "kernel" << std::endl;
	Dump(B);
	std::cout << "result of Max2D()" << std::endl;
	Dump(result);

	BOOST_REQUIRE(result.At(0) == 6);
	BOOST_REQUIRE(result.At(19) == 9);

	B.Set({0, 0, 0, 0, 1, 0, 0, 0, 0}); // should return A

	result = numerical_functions::Max2D(A, B);

	for(size_t i=0; i < A.Size(); ++i)
	{
		BOOST_REQUIRE(A.At(i) == result.At(i));
	}
}
