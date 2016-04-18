#include "himan_unit.h"
#include "numerical_functions.h"

#include "numerical_functions_helper.h"

#include "timer.h"

#define BOOST_TEST_MODULE numerical_functions

using namespace std;
using namespace himan;

const double kEpsilon = 1e-3;

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
	for (size_t i=0; i < C.SizeX();++i){
    		for (size_t j=0; j < C.SizeY();++j){
      			std::cout << C.At(i,j,0) << " ";
    		}
    		std::cout << std::endl;
  	}

	std::cout << std::endl << "Matrix D as reference case for Filter2D computation:" << std::endl; 

	for (size_t i=0; i < D.SizeX();++i){
    		for (size_t j=0; j < D.SizeY();++j){
      			std::cout << D.At(i,j,0) << " ";
    		}
    		std::cout << std::endl;
  	}

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
