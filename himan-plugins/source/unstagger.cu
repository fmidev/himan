/**
 * @file unstagger.cpp
 *
 * Calculate the co-located velocity field for U and V
 *

  Description:

  The grid in a staggered arrangement consists of central points ('o') where most parameter values
  e.g. temperature and pressure are stored. The u-velocity points ('u') are shifted east by a half
  grid spacing and v-velocity points ('v') shifted south by a half grid spacing. A example on a 3x3
  grid is given below.

  o---u---o---u---o---u
  |       |       |
  v       v       v
  |       |       |
  o---u---o---u---o---u
  |       |       |
  v       v       v
  |       |       |
  o---u---o---u---o---u
  |       |       |
  v       v       v

  If collocation of all parameters is required, u- and v-velocity needs to be unstaggered. The
  unstaggering is done by solving a set of linear equations. For the example grid shown above
  the linear equations are.

  Case u:                                 Case v:

    1*u(1,1)              = o(1,1)          1*v(1,1)              = o(1,1)
  0.5*u(1,1) + 0.5*u(1,2) = o(1,2)          1*v(1,2)              = o(1,2)
  0.5*u(1,2) + 0.5*u(1,3) = o(1,3)          1*v(1,3)              = o(1,3)
    1*u(2,1)              = o(2,1)        0.5*v(1,1) + 0.5*v(2,1) = o(2,1)
  0.5*u(2,1) + 0.5*u(2,2) = o(2,2)        0.5*v(1,2) + 0.5*v(2,2) = o(2,2)
  0.5*u(2,2) + 0.5*u(2,3) = o(2,3)        0.5*v(1,3) + 0.5*v(2,3) = o(2,3)
    1*u(3,1)              = o(3,1)        0.5*v(2,1) + 0.5*v(3,1) = o(3,1)
  0.5*u(3,1) + 0.5*u(3,2) = o(3,2)        0.5*v(2,2) + 0.5*v(3,2) = o(3,2)
  0.5*u(3,2) + 0.5*u(3,3) = o(3,3)        0.5*v(2,3) + 0.5*v(3,3) = o(3,3)

  These equations can be re-written in matrix-vector form

                   |-      -|
                   | u(1,1) |
                   |   .    |
                   |   .    |
                   | u(3,3) |
                   |-      -|

  |-           -|  |-      -|
  | w11 ... w19 |  | o(1,1) |
  | .       .   |  |   .    |
  | .       .   |  |   .    |
  | w91 ... w99 |  | o(3,3) |
  |-           -|  |-      -|

  where the weighting matrices for the u- and v-case are diagonal matrices.

  The diagonal matrix to unstagger the u-velocity field has the form

        | 2 0 0 0 0 0 0 0 0 |
        | 1 1 0 0 0 0 0 0 0 |
        | 0 1 1 0 0 0 0 0 0 |
        | 0 0 0 2 0 0 0 0 0 |
  1/2 * | 0 0 0 1 1 0 0 0 0 |  = U_unstagger
        | 0 0 0 0 1 1 0 0 0 |
        | 0 0 0 0 0 0 2 0 0 |
        | 0 0 0 0 0 0 1 1 0 |
        | 0 0 0 0 0 0 0 1 1 |

  and the diagonal matrix to unstagger the v-velocity

        | 2 0 0 0 0 0 0 0 0 |
        | 0 2 0 0 0 0 0 0 0 |
        | 0 0 2 0 0 0 0 0 0 |
        | 1 0 0 1 0 0 0 0 0 |
  1/2 * | 0 1 0 0 1 0 0 0 0 |  = V_unstagger
        | 0 0 1 0 0 1 0 0 0 |
        | 0 0 0 1 0 0 1 0 0 |
        | 0 0 0 0 1 0 0 1 0 |
        | 0 0 0 0 0 1 0 0 1 |

  In this form the problem can be solved efficiently using a sparse matrix library
  matrix-vector multiplication. In this implementation CUSP is used for that purpose.

  **/

#include "cuda_plugin_helper.h"
#include "unstagger.cuh"
#ifdef DEBUG
#undef DEBUG
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/dia_matrix.h>
#include <cusp/multiply.h>
#include <thrust/system/cuda/execution_policy.h>
#define DEBUG
#else
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/dia_matrix.h>
#include <cusp/multiply.h>
#include <thrust/system/cuda/execution_policy.h>
#endif

cusp::dia_matrix<int, double, cusp::device_memory> U_unstagger;
cusp::dia_matrix<int, double, cusp::device_memory> V_unstagger;

void himan::plugin::unstagger_cuda::Init(size_t NX, size_t NY)
{
	// create diagonal matix with constant coefficiants
	size_t N = NX * NY;
	cusp::dia_matrix<int, double, cusp::host_memory> h_U_unstagger(N, N, 2 * N, 2);
	cusp::dia_matrix<int, double, cusp::host_memory> h_V_unstagger(N, N, 2 * N, 2);

	cusp::array2d<double, cusp::device_memory> Diags(N, 2, 0.5);

	h_U_unstagger.diagonal_offsets[0] = 0;
	h_U_unstagger.diagonal_offsets[1] = -1;

	h_U_unstagger.values = Diags;

	// alter coefficient for interpolation of first column in U
	for (size_t i = 0; i < NY; ++i)
	{
		h_U_unstagger.values(i * NX, 0) = 1.0;
		h_U_unstagger.values(i * NX, 1) = 0.0;
	}

	h_V_unstagger.diagonal_offsets[0] = 0;
	h_V_unstagger.diagonal_offsets[1] = -NX;

	h_V_unstagger.values = Diags;

	// alter coefficient for interpolation of first row in V
	for (size_t i = 0; i < NX; ++i)
	{
		h_V_unstagger.values(i, 0) = 1.0;
		h_V_unstagger.values(i, 1) = 0.0;
	}

	// copy matrices to device
	U_unstagger = h_U_unstagger;
	V_unstagger = h_V_unstagger;
}

std::pair<std::vector<double>, std::vector<double>> himan::plugin::unstagger_cuda::Process(std::vector<double>& U_in,
                                                                                           std::vector<double>& V_in)
{
	size_t N = U_in.size();

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	std::vector<double> U_out(N);
	std::vector<double> V_out(N);

	CUDA_CHECK(cudaHostRegister(U_in.data(), sizeof(double) * N, 0));
	CUDA_CHECK(cudaHostRegister(V_in.data(), sizeof(double) * N, 0));
	CUDA_CHECK(cudaHostRegister(U_out.data(), sizeof(double) * N, 0));
	CUDA_CHECK(cudaHostRegister(V_out.data(), sizeof(double) * N, 0));

	// create 1d arrays on device
	double* d_U = nullptr;      // pointer to device memory pointing to incoming data of U
	double* d_V = nullptr;      // pointer to device memory pointing to incoming data of V
	double* d_U_out = nullptr;  // pointer to device memory to unstaggered data of U
	double* d_V_out = nullptr;  // pointer to device memory to unstaggered data of V

	// allocate memory
	CUDA_CHECK(cudaMalloc((void**)&d_U, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((void**)&d_V, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((void**)&d_U_out, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc((void**)&d_V_out, sizeof(double) * N));

	// copy data to device
	CUDA_CHECK(cudaMemcpyAsync(d_U, U_in.data(), sizeof(double) * N, cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_V, V_in.data(), sizeof(double) * N, cudaMemcpyHostToDevice, stream));

	// cast raw pointer to thrust device pointer
	thrust::device_ptr<double> dt_U = thrust::device_pointer_cast(d_U);
	thrust::device_ptr<double> dt_V = thrust::device_pointer_cast(d_V);
	thrust::device_ptr<double> dt_U_out = thrust::device_pointer_cast(d_U_out);
	thrust::device_ptr<double> dt_V_out = thrust::device_pointer_cast(d_V_out);

	// create cusp::array1d
	auto U_device = cusp::array1d_view<thrust::device_ptr<double>>(dt_U, dt_U + N);
	auto V_device = cusp::array1d_view<thrust::device_ptr<double>>(dt_V, dt_V + N);
	auto U_device_out = cusp::array1d_view<thrust::device_ptr<double>>(dt_U_out, dt_U_out + N);
	auto V_device_out = cusp::array1d_view<thrust::device_ptr<double>>(dt_V_out, dt_V_out + N);

	// perform the unstagger operation
	cusp::multiply(thrust::cuda::par.on(stream), U_unstagger, U_device, U_device_out);
	cusp::multiply(thrust::cuda::par.on(stream), V_unstagger, V_device, V_device_out);

	// copy result back to host
	CUDA_CHECK(cudaMemcpyAsync(U_out.data(), d_U_out, sizeof(double) * N, cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(V_out.data(), d_V_out, sizeof(double) * N, cudaMemcpyDeviceToHost, stream));

	// free memory
	CUDA_CHECK(cudaFree(d_U));
	CUDA_CHECK(cudaFree(d_V));
	CUDA_CHECK(cudaFree(d_U_out));
	CUDA_CHECK(cudaFree(d_V_out));

	CUDA_CHECK(cudaHostUnregister(U_in.data()));
	CUDA_CHECK(cudaHostUnregister(V_in.data()));
	CUDA_CHECK(cudaHostUnregister(U_out.data()));
	CUDA_CHECK(cudaHostUnregister(V_out.data()));

	CUDA_CHECK(cudaStreamDestroy(stream));

	return std::make_pair(U_out, V_out);
}
