#ifndef PBA2D_EDT_API_HPP
#define PBA2D_EDT_API_HPP

#include <cuda_runtime.h>

// Function declarations for PBA-based EDT API
void pba_edt_2d(char* d_boundary, int* index, float* distance, 
                unsigned int width, unsigned int height);

void pba_edt_2d_optimized(char* d_boundary, int* index, float* distance, 
                          unsigned int size);

// Proper CUDA EDT implementation for comparison
void edt_2d(char* d_boundary, int* index, float* distance, 
            unsigned int width, unsigned int height);

void brute_force_edt_2d(char* d_boundary, int* index, float* distance,
                        unsigned int width, unsigned int height);

void benchmark_pba_vs_edt_2d(char* d_boundary, unsigned int width, unsigned int height, 
                             int num_iterations = 10);

#endif // PBA2D_EDT_API_HPP
