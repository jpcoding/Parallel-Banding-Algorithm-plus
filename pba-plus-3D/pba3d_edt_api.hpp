#ifndef PBA3D_EDT_API_HPP
#define PBA3D_EDT_API_HPP

#include <cuda_runtime.h>

// Function declarations for PBA-based EDT API
void pba_edt_3d(char* d_boundary, int* index, float* distance, 
                unsigned int width, unsigned int height, unsigned int depth);

void pba_edt_3d_optimized(char* d_boundary, int* index, float* distance, 
                          unsigned int size);

// Proper CUDA EDT implementation for comparison
void edt_3d(char* d_boundary, int* index, float* distance, 
            unsigned int width, unsigned int height, unsigned int depth);

void brute_force_edt_3d(char* d_boundary, int* index, float* distance,
                        unsigned int width, unsigned int height, unsigned int depth);

void benchmark_pba_vs_edt(char* d_boundary, unsigned int width, unsigned int height, unsigned int depth, 
                         int num_iterations = 10);

#endif // PBA3D_EDT_API_HPP
