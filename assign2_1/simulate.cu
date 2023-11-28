/*
 * simulate.cu
 *
 * Implementation of a wave equation simulation, parallelized on the GPU using
 * CUDA.
 *
 * You are supposed to edit this file with your implementation, and this file
 * only.
 *
 */

#include <cstdlib>
#include <iostream>

#include "simulate.hh"

using namespace std;

// Function to simulate the wave equation using CUDA
__global__ void waveEquationKernel(const long i_max, double *old_array, double *current_array, double *next_array)
{
    // Calculate the index for the current thread
    long i = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread is within the valid range of indices
    if (i > 0 && i < i_max - 1)
    {
        // Compute the new wave amplitude using the given wave equation
        next_array[i] = 2 * current_array[i] - old_array[i] +
                        0.15 * (old_array[i - 1] - 2 * current_array[i] + old_array[i + 1]);
    }

    // Synchronize threads to ensure all computations are done before moving to the next time step
    __syncthreads();

    // Swap arrays for the next time step
    if (i > 0 && i < i_max - 1)
    {
        old_array[i] = current_array[i];
        current_array[i] = next_array[i];
    }
}

/* Utility function, use to do error checking for CUDA calls
 *
 * Use this function like this:
 *     checkCudaCall(<cuda_call>);
 *
 * For example:
 *     checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));
 *
 * Special case to check the result of the last kernel invocation:
 *     kernel<<<...>>>(...);
 *     checkCudaCall(cudaGetLastError());
 **/
static void checkCudaCall(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(EXIT_FAILURE);
    }
}

/* Function that will simulate the wave equation, parallelized using CUDA.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * num_threads: how many threads to use (excluding the main threads)
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 *
 */
double *simulate(const long i_max, const long t_max, const long block_size,
                 double *old_array, double *current_array, double *next_array)
{
    // Allocate GPU memory
    double *d_old_array, *d_current_array, *d_next_array;
    checkCudaCall(cudaMalloc((void **)&d_old_array, i_max * sizeof(double)));
    checkCudaCall(cudaMalloc((void **)&d_current_array, i_max * sizeof(double)));
    checkCudaCall(cudaMalloc((void **)&d_next_array, i_max * sizeof(double)));

    // Copy data from CPU to GPU
    checkCudaCall(cudaMemcpy(d_old_array, old_array, i_max * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(d_current_array, current_array, i_max * sizeof(double), cudaMemcpyHostToDevice));

    // Define the grid and block dimensions
    dim3 gridDim((i_max + block_size - 1) / block_size, 1, 1);
    dim3 blockDim(block_size, 1, 1);

    // Iterate over time steps
    for (long t = 0; t < t_max; ++t)
    {
        // Launch the kernel to compute the wave equation
        waveEquationKernel<<<gridDim, blockDim>>>(i_max, d_old_array, d_current_array, d_next_array);

        // Check for CUDA errors after launching the kernel
        checkCudaCall(cudaGetLastError());
        checkCudaCall(cudaDeviceSynchronize());
    }

    // Copy the final result back from GPU to CPU
    checkCudaCall(cudaMemcpy(current_array, d_current_array, i_max * sizeof(double), cudaMemcpyDeviceToHost));

    // Free GPU memory
    checkCudaCall(cudaFree(d_old_array));
    checkCudaCall(cudaFree(d_current_array));
    checkCudaCall(cudaFree(d_next_array));

    return current_array;
}