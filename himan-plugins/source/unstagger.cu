#include "unstagger.cuh"
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/multiply.h>
#include <cusp/dia_matrix.h>
#include <thrust/system/cuda/execution_policy.h>
#include "cuda_plugin_helper.h"

cusp::dia_matrix<int,double,cusp::device_memory> U_unstagger;
cusp::dia_matrix<int,double,cusp::device_memory> V_unstagger;

void himan::plugin::unstagger_cuda::Init(size_t NX, size_t NY)
{
        // create diagonal matix with constant coefficiants
	size_t N = NX*NY;
	cusp::dia_matrix<int,double,cusp::host_memory> h_U_unstagger(N,N,2*N,2);
	cusp::dia_matrix<int,double,cusp::host_memory> h_V_unstagger(N,N,2*N,2);

	cusp::array2d<double,cusp::device_memory> Diags(N,2,0.5);

        h_U_unstagger.diagonal_offsets[0] = 0;
        h_U_unstagger.diagonal_offsets[1] = -1;

        h_U_unstagger.values = Diags;

        // alter coefficient for interpolation of first column in U
        for (size_t i=0; i<NY; ++i)
        {
            h_U_unstagger.values(i*NX,0) = 1.0;
            h_U_unstagger.values(i*NX,1) = 0.0;

        }

        h_V_unstagger.diagonal_offsets[0] = 0;
        h_V_unstagger.diagonal_offsets[1] = -NX;

        h_V_unstagger.values = Diags;

        // alter coefficient for interpolation of first row in V
        for (size_t i=0; i<NX; ++i)
        {
            h_V_unstagger.values(i,0) = 1.0;
            h_V_unstagger.values(i,1) = 0.0;

        }

        // copy matrices to device
 	U_unstagger = h_U_unstagger;
	V_unstagger = h_V_unstagger;
}

std::pair<std::vector<double>, std::vector<double>> himan::plugin::unstagger_cuda::Process(std::vector<double> &U_in, std::vector<double> &V_in)
{
	size_t N = U_in.size();

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        std::vector<double> U_out(N);
        std::vector<double> V_out(N);

        CUDA_CHECK(cudaHostRegister(U_in.data(),sizeof(double) * N,0));
        CUDA_CHECK(cudaHostRegister(V_in.data(),sizeof(double) * N,0));
        CUDA_CHECK(cudaHostRegister(U_out.data(),sizeof(double) * N,0));
        CUDA_CHECK(cudaHostRegister(V_out.data(),sizeof(double) * N,0));

        // create 1d arrays on device
        double* d_U = 0; // pointer to device memory pointing to incoming data of U
        double* d_V = 0; // pointer to device memory pointing to incoming data of V
        double* d_U_out = 0; // pointer to device memory to unstaggered data of U
        double* d_V_out = 0; // pointer to device memory to unstaggered data of V

	// allocate memory
        CUDA_CHECK(cudaMalloc((double**) &d_U, sizeof(double) * N));
        CUDA_CHECK(cudaMalloc((double**) &d_V, sizeof(double) * N));
        CUDA_CHECK(cudaMalloc((double**) &d_U_out, sizeof(double) * N));
        CUDA_CHECK(cudaMalloc((double**) &d_V_out, sizeof(double) * N));

	// copy data to device
        CUDA_CHECK(cudaMemcpyAsync(d_U,U_in.data(),sizeof(double) * N, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_V,V_in.data(),sizeof(double) * N, cudaMemcpyHostToDevice, stream));

	// cast raw pointer to thrust device pointer 
        thrust::device_ptr<double> dt_U = thrust::device_pointer_cast(d_U);
        thrust::device_ptr<double> dt_V = thrust::device_pointer_cast(d_V); 
        thrust::device_ptr<double> dt_U_out = thrust::device_pointer_cast(d_U_out); 
        thrust::device_ptr<double> dt_V_out = thrust::device_pointer_cast(d_V_out); 

	// create cusp::array1d
        auto U_device = cusp::array1d_view<thrust::device_ptr<double>> (dt_U, dt_U + N); // create array1d
        auto V_device = cusp::array1d_view<thrust::device_ptr<double>> (dt_V, dt_V + N); // create array1d
        auto U_device_out = cusp::array1d_view<thrust::device_ptr<double>> (dt_U_out, dt_U_out + N); // create array1d
        auto V_device_out = cusp::array1d_view<thrust::device_ptr<double>> (dt_V_out, dt_V_out + N); // create array1d

	// perform the unstagger operation
	cusp::multiply(thrust::cuda::par.on(stream),U_unstagger,U_device,U_device_out);
	cusp::multiply(thrust::cuda::par.on(stream),V_unstagger,V_device,V_device_out);

	// copy result back to host
        CUDA_CHECK(cudaMemcpyAsync(U_out.data(),d_U_out,sizeof(double) * N, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(V_out.data(),d_V_out,sizeof(double) * N, cudaMemcpyDeviceToHost, stream));

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

