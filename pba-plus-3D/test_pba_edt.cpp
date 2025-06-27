#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include "pba3d_edt_api.hpp"

// Helper function to write binary data
template <typename Type>
void write_binary_file(const char* filename, Type* data, size_t num_elements) {
    FILE* file = fopen(filename, "wb");
    if (file) {
        fwrite(data, sizeof(Type), num_elements, file);
        fclose(file);
        printf("Wrote %zu elements to %s\n", num_elements, filename);
    }
}

// Generate random boundary points
void generate_random_boundary(char* boundary, int width, int height, int depth, int num_points) {
    // Initialize all to 0 (non-boundary)
    size_t size = width * height * depth;
    memset(boundary, 0, size);
    
    // Set random points as boundary (value 1)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> x_dist(0, width - 1);
    std::uniform_int_distribution<> y_dist(0, height - 1);
    std::uniform_int_distribution<> z_dist(0, depth - 1);
    
    for (int i = 0; i < num_points; i++) {
        int x = x_dist(gen);
        int y = y_dist(gen);
        int z = z_dist(gen);
        int idx = z * width * height + y * width + x;
        boundary[idx] = 1;
    }
    
    printf("Generated %d random boundary points in %dx%dx%d volume\n", 
           num_points, width, height, depth);
}

int main() {
    printf("=== PBA-based EDT API Test ===\n");
    
    // Test parameters  
    const int width = 256;  // Standard comparison size
    const int height = 256; 
    const int depth = 256;
    const int num_boundary_points = 100;
    
    size_t size = width * height * depth;
    
    // Allocate host memory
    std::vector<char> h_boundary(size, 0);
    std::vector<int> h_index(size * 3, 0);
    std::vector<float> h_distance(size, 0.0f);
    
    // Generate random boundary points
    generate_random_boundary(h_boundary.data(), width, height, depth, num_boundary_points);
    
    // Allocate device memory
    char* d_boundary;
    int* d_index;
    float* d_distance;
    
    cudaMalloc(&d_boundary, size * sizeof(char));
    cudaMalloc(&d_index, size * 3 * sizeof(int));
    cudaMalloc(&d_distance, size * sizeof(float));
    
    // Copy boundary data to device
    cudaMemcpy(d_boundary, h_boundary.data(), size * sizeof(char), cudaMemcpyHostToDevice);
    
    printf("\nRunning PBA-based EDT computation...\n");
    
    // Test the optimized version (assumes power-of-2 cubic dimensions)
    if (width == height && height == depth && (width & (width - 1)) == 0) {
        printf("Using optimized cubic version\n");
        pba_edt_3d_optimized(d_boundary, d_index, d_distance, width);
    } else {
        printf("Using general version\n");
        pba_edt_3d(d_boundary, d_index, d_distance, width, height, depth);
    }
    
    printf("PBA EDT computation completed\n");
    
    // Copy results back to host
    cudaMemcpy(h_index.data(), d_index, size * 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_distance.data(), d_distance, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Analyze results
    printf("\n=== Results Analysis ===\n");
    
    // Find min/max distances
    float min_dist = h_distance[0];
    float max_dist = h_distance[0];
    double avg_dist = 0.0;
    int zero_dist_count = 0;
    
    for (size_t i = 0; i < size; i++) {
        float dist = h_distance[i];
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
        avg_dist += dist;
        if (dist == 0.0f) zero_dist_count++;
    }
    avg_dist /= size;
    
    printf("Distance statistics:\n");
    printf("  Min distance: %.3f\n", min_dist);
    printf("  Max distance: %.3f\n", max_dist);
    printf("  Average distance: %.3f\n", avg_dist);
    printf("  Zero distance points: %d (boundary points)\n", zero_dist_count);
    
    // Sample some results
    printf("\nSample results:\n");
    for (int i = 0; i < 10 && i * 10000 < size; i++) {
        int idx = i * 10000;
        int z = idx / (width * height);
        int y = (idx % (width * height)) / width;
        int x = idx % width;
        
        printf("  Point (%d,%d,%d): distance=%.3f, nearest=(%d,%d,%d)\n",
               x, y, z, h_distance[idx],
               h_index[idx * 3 + 2], h_index[idx * 3 + 1], h_index[idx * 3 + 0]);
    }
    
    // Save results to files
    write_binary_file("pba_distances.dat", h_distance.data(), size);
    write_binary_file("pba_indices.dat", h_index.data(), size * 3);
    write_binary_file("boundary_mask.dat", h_boundary.data(), size);
    
    // Benchmark
    printf("\n=== Performance Benchmark ===\n");
    benchmark_pba_vs_edt(d_boundary, width, height, depth, 5);
    
    // Clean up
    cudaFree(d_boundary);
    cudaFree(d_index);
    cudaFree(d_distance);
    
    printf("\n=== Test completed successfully ===\n");
    return 0;
}
