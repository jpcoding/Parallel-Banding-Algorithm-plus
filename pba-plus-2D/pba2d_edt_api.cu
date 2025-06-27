#include "pba2d_edt_api.hpp"
#include "pba/pba2D.h"
#include <algorithm>
#include <cstring>
#include <stdio.h>
#include <float.h>

// CUDA kernel to initialize padded PBA array with MARKER values
__global__ void init_padded_pba_2d(short* pba_input, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= size || y >= size) return;
    
    int idx = y * size + x;
    pba_input[idx * 2] = MARKER;
    pba_input[idx * 2 + 1] = MARKER;
}

// CUDA kernel to convert boundary mask to PBA input format
__global__ void init_pba_2d(char* boundary, short* pba_input, char b_tag, 
                            int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    if (boundary[idx] == b_tag) {
        // This is a seed point - store coordinates
        pba_input[idx * 2] = (short)x;
        pba_input[idx * 2 + 1] = (short)y;
    } else {
        // Not a seed point - use marker
        pba_input[idx * 2] = MARKER;
        pba_input[idx * 2 + 1] = MARKER;
    }
}

// CUDA kernel to convert PBA output to index format and calculate distances
__global__ void pba_to_results_2d(short* pba_output, int* index, float* distance,
                                  int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Get nearest site coordinates from PBA output
    short nearest_x = pba_output[idx * 2];
    short nearest_y = pba_output[idx * 2 + 1];
    
    // Store in index array (y, x format to match common EDT conventions)
    index[idx * 2 + 0] = nearest_y;
    index[idx * 2 + 1] = nearest_x;
    
    // Calculate distance
    float dx = nearest_x - x;
    float dy = nearest_y - y;
    distance[idx] = sqrtf(dx * dx + dy * dy);
}

// PBA-based EDT implementation with same API as a standard edt_2d
void pba_edt_2d(char* d_boundary, int* index, float* distance, 
                unsigned int width, unsigned int height) {
    
    size_t size = width * height;
    
    // Allocate temporary arrays for PBA (PBA uses short pairs)
    short* d_pba_input;
    short* d_pba_output;
    
    cudaMalloc(&d_pba_input, size * 2 * sizeof(short));
    cudaMalloc(&d_pba_output, size * 2 * sizeof(short));
    
    // Set up grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, 
              (height + block.y - 1) / block.y);
    
    // Convert boundary mask to PBA input format
    init_pba_2d<<<grid, block>>>(d_boundary, d_pba_input, (char)1, width, height);
    cudaDeviceSynchronize();
    
    // Initialize PBA with the texture size (must be power of 2)
    int texture_size = std::max(width, height);
    // Round up to next power of 2
    int pba_size = 64; // Minimum size for 2D (2^6)
    while (pba_size < texture_size) {
        pba_size *= 2;
    }
    
    // If dimensions don't match PBA requirements, we need to pad
    if (width == height && width == pba_size) {
        // Perfect match - use PBA directly
        pba2DInitialization(pba_size, 1);
        pba2DVoronoiDiagram(d_pba_input, d_pba_output, 1, 1, 2);
        pba2DDeinitialization();
    } else {
        // Need to handle non-square or non-power-of-2 dimensions
        // Allocate padded arrays
        short* d_pba_padded_input;
        short* d_pba_padded_output;
        size_t padded_size = pba_size * pba_size;
        
        cudaMalloc(&d_pba_padded_input, padded_size * 2 * sizeof(short));
        cudaMalloc(&d_pba_padded_output, padded_size * 2 * sizeof(short));
        
        // Initialize padded arrays with MARKER using a kernel
        dim3 padded_block(16, 16);
        dim3 padded_grid((pba_size + padded_block.x - 1) / padded_block.x, 
                        (pba_size + padded_block.y - 1) / padded_block.y);
        
        init_padded_pba_2d<<<padded_grid, padded_block>>>(d_pba_padded_input, pba_size);
        cudaDeviceSynchronize();
        
        // Copy original data to padded array
        if (width <= pba_size && height <= pba_size) {
            for (int y = 0; y < height; y++) {
                size_t src_offset = y * width * 2;
                size_t dst_offset = y * pba_size * 2;
                cudaMemcpy(d_pba_padded_input + dst_offset, 
                          d_pba_input + src_offset, 
                          width * 2 * sizeof(short), cudaMemcpyDeviceToDevice);
            }
        }
        
        // Run PBA
        pba2DInitialization(pba_size, 1);
        pba2DVoronoiDiagram(d_pba_padded_input, d_pba_padded_output, 1, 1, 2);
        pba2DDeinitialization();
        
        // Copy results back
        for (int y = 0; y < height; y++) {
            size_t src_offset = y * pba_size * 2;
            size_t dst_offset = y * width * 2;
            cudaMemcpy(d_pba_output + dst_offset,
                      d_pba_padded_output + src_offset,
                      width * 2 * sizeof(short), cudaMemcpyDeviceToDevice);
        }
        
        cudaFree(d_pba_padded_input);
        cudaFree(d_pba_padded_output);
    }
    
    // Convert PBA output to your format and calculate distances
    pba_to_results_2d<<<grid, block>>>(d_pba_output, index, distance, width, height);
    cudaDeviceSynchronize();
    
    // Clean up
    cudaFree(d_pba_input);
    cudaFree(d_pba_output);
}

// Optimized version that assumes square power-of-2 dimensions
void pba_edt_2d_optimized(char* d_boundary, int* index, float* distance, 
                          unsigned int size) {
    // This version assumes size is power of 2 and width=height=size
    
    size_t total_elements = size * size;
    
    // Allocate temporary arrays for PBA
    short* d_pba_input;
    short* d_pba_output;
    
    cudaMalloc(&d_pba_input, total_elements * 2 * sizeof(short));
    cudaMalloc(&d_pba_output, total_elements * 2 * sizeof(short));
    
    // Set up grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((size + block.x - 1) / block.x, 
              (size + block.y - 1) / block.y);
    
    // Convert boundary mask to PBA input format
    init_pba_2d<<<grid, block>>>(d_boundary, d_pba_input, (char)1, size, size);
    cudaDeviceSynchronize();
    
    // Run PBA
    pba2DInitialization(size, 1);
    pba2DVoronoiDiagram(d_pba_input, d_pba_output, 1, 1, 2);
    pba2DDeinitialization();
    
    // Convert PBA output to your format and calculate distances
    pba_to_results_2d<<<grid, block>>>(d_pba_output, index, distance, size, size);
    cudaDeviceSynchronize();
    
    // Clean up
    cudaFree(d_pba_input);
    cudaFree(d_pba_output);
}

// Simple CUDA EDT implementation for comparison
__global__ void edt_kernel_2d(char* boundary, int* index, float* distance,
                              int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Initialize with invalid values to catch bugs
    float min_dist = 1000000.0f;
    int nearest_x = -1, nearest_y = -1;
    bool found_boundary = false;
    
    // Brute force search for nearest boundary pixel
    for (int by = 0; by < height; by++) {
        for (int bx = 0; bx < width; bx++) {
            int b_idx = by * width + bx;
            if (boundary[b_idx] == 1) {
                found_boundary = true;
                float dx = (float)(bx - x);
                float dy = (float)(by - y);
                float dist = sqrtf(dx * dx + dy * dy);
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest_x = bx;
                    nearest_y = by;
                }
            }
        }
    }
    
    // If no boundary found, something is very wrong
    if (!found_boundary) {
        distance[idx] = -1.0f;  // Error indicator
        index[idx * 2 + 0] = -1;
        index[idx * 2 + 1] = -1;
    } else {
        distance[idx] = min_dist;
        index[idx * 2 + 0] = nearest_y;
        index[idx * 2 + 1] = nearest_x;
    }
}

// Proper CUDA EDT implementation for comparison
void edt_2d(char* d_boundary, int* index, float* distance, 
            unsigned int width, unsigned int height) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, 
              (height + block.y - 1) / block.y);
    
    edt_kernel_2d<<<grid, block>>>(d_boundary, index, distance, width, height);
    cudaDeviceSynchronize();
}

// Brute force EDT implementation (same as edt_2d for simplicity)
void brute_force_edt_2d(char* d_boundary, int* index, float* distance,
                        unsigned int width, unsigned int height) {
    edt_2d(d_boundary, index, distance, width, height);
}

// Benchmark function
void benchmark_pba_vs_edt_2d(char* d_boundary, unsigned int width, unsigned int height, 
                             int num_iterations) {
    size_t size = width * height;
    
    // Allocate result arrays
    int* index_pba;
    float* distance_pba;
    int* index_edt;
    float* distance_edt;
    
    cudaMalloc(&index_pba, size * 2 * sizeof(int));
    cudaMalloc(&distance_pba, size * sizeof(float));
    cudaMalloc(&index_edt, size * 2 * sizeof(int));
    cudaMalloc(&distance_edt, size * sizeof(float));
    
    printf("Benchmarking PBA vs EDT for %dx%d image with %d iterations\n", 
           width, height, num_iterations);
    
    // Time PBA approach
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        pba_edt_2d(d_boundary, index_pba, distance_pba, width, height);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float pba_time = 0;
    cudaEventElapsedTime(&pba_time, start, stop);
    
    // Time EDT approach
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        edt_2d(d_boundary, index_edt, distance_edt, width, height);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float edt_time = 0;
    cudaEventElapsedTime(&edt_time, start, stop);
    
    printf("PBA average time: %.3f ms\n", pba_time / num_iterations);
    printf("EDT average time: %.3f ms\n", edt_time / num_iterations);
    printf("Speedup: %.2fx\n", edt_time / pba_time);
    
    // Clean up
    cudaFree(index_pba);
    cudaFree(distance_pba);
    cudaFree(index_edt);
    cudaFree(distance_edt);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
