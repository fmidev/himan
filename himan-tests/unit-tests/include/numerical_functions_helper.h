#ifndef NUMERICAL_FUNCTIONS_HELPER_H
#define NUMERICAL_FUNCTIONS_HELPER_H

#include "himan_common.h"

// Setup for Filter2D tests
static void FilterTestSetup(himan::matrix<double>& A, himan::matrix<double>& B, himan::matrix<double>& D)
{
    // Fill matrix A that will be smoothened with checker-board pattern
    for(size_t i=0; i < A.Size(); ++i)
    {
        if(i % 2 == 0)
        {
            A.Set(i, 0);
        }
        else
        {
            A.Set(i, 36);
        }
    }
    
    // Fill matrix D with solution of the smoothened matrix Filter2D(A,B) 
    for(size_t i=0; i < D.SizeX(); ++i)
    {
        for(size_t j=0; j < D.SizeY(); ++j)
        {
            if(i == 0 || i == A.SizeX()-1 || j == 0 || j == A.SizeY()-1)
            {
                D.Set(i,j,0,18);
            }
            else if ((i % 2 != 0 && j % 2 != 0) || (i % 2 == 0 && j % 2 == 0))
            {
                D.Set(i,j,0,16);
            }
            else
            {
                D.Set(i,j,0,20);
            }
        }
    }

    // Fill matrix B (filter kernel) with constant values 1/9 so that sum(B) == 1
    double filter_coeff(1.0/9.0);
    B.Fill(filter_coeff);
}

// NUMERICAL_FUNCTIONS_HELPER_H
#endif
