#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "pba2d_edt_api.hpp"

int main() {
    printf("=== Simple Debug Test ===\n");
    
    // Very simple 4x4 test case
    const int width = 4;
    const int height = 4;
    size_t image_size = width * height;
    size_t result_size = image_size * 2;
    
    // Host arrays
    char* h_boundary = (char*)malloc(image_size);
    int* h_index = (int*)malloc(result_size * sizeof(int));
    float* h_distance = (float*)malloc(image_size * sizeof(float));
    
    // Create simple boundary pattern
    memset(h_boundary, 0, image_size);
    h_boundary[0] = 1;  // (0,0)
    h_boundary[15] = 1; // (3,3)
    
    printf("Boundary pattern (4x4):\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%d ", h_boundary[y * width + x]);
        }
        printf("\n");
    }
    
    // Device arrays
    char* d_boundary;
    int* d_index;
    float* d_distance;
    
    cudaMalloc(&d_boundary, image_size);
    cudaMalloc(&d_index, result_size * sizeof(int));
    cudaMalloc(&d_distance, image_size * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_boundary, h_boundary, image_size, cudaMemcpyHostToDevice);
    
    // Test brute force EDT
    printf("\n=== Testing brute force EDT ===\n");
    edt_2d(d_boundary, d_index, d_distance, width, height);
    
    // Copy results back
    cudaMemcpy(h_index, d_index, result_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_distance, d_distance, image_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("EDT Results:\n");
    printf("Distances:\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            printf("%.2f ", h_distance[idx]);
        }
        printf("\n");
    }
    
    printf("Nearest indices (y,x):\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            printf("(%d,%d) ", h_index[idx * 2], h_index[idx * 2 + 1]);
        }
        printf("\n");
    }
    
    // Test PBA EDT
    printf("\n=== Testing PBA EDT ===\n");
    pba_edt_2d(d_boundary, d_index, d_distance, width, height);
    
    // Copy results back
    cudaMemcpy(h_index, d_index, result_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_distance, d_distance, image_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("PBA Results:\n");
    printf("Distances:\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            printf("%.2f ", h_distance[idx]);
        }
        printf("\n");
    }
    
    printf("Nearest indices (y,x):\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            printf("(%d,%d) ", h_index[idx * 2], h_index[idx * 2 + 1]);
        }
        printf("\n");
    }
    
    // Clean up
    free(h_boundary);
    free(h_index);
    free(h_distance);
    cudaFree(d_boundary);
    cudaFree(d_index);
    cudaFree(d_distance);
    
    return 0;
}
