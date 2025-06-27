#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include "pba2d_edt_api.hpp"

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
void generate_random_boundary(char* boundary, int width, int height, int num_points) {
    // Initialize all to 0 (non-boundary)
    size_t size = width * height;
    memset(boundary, 0, size);
    
    // Set random points as boundary (value 1)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> x_dist(0, width - 1);
    std::uniform_int_distribution<> y_dist(0, height - 1);
    
    for (int i = 0; i < num_points; i++) {
        int x = x_dist(gen);
        int y = y_dist(gen);
        int idx = y * width + x;
        boundary[idx] = 1;
    }
    
    printf("Generated %d random boundary points in %dx%d image\n", 
           num_points, width, height);
}

// Generate a simple test pattern
void generate_test_pattern(char* boundary, int width, int height) {
    memset(boundary, 0, width * height);
    
    // Add a few specific points for testing
    // Center point
    int cx = width / 2;
    int cy = height / 2;
    boundary[cy * width + cx] = 1;
    
    // Corner points
    boundary[0] = 1;  // top-left
    boundary[width - 1] = 1;  // top-right
    boundary[(height - 1) * width] = 1;  // bottom-left
    boundary[(height - 1) * width + width - 1] = 1;  // bottom-right
    
    // Add some additional points for better testing
    // Points along edges
    boundary[cy * width + 10] = 1;  // left edge
    boundary[cy * width + width - 10] = 1;  // right edge
    boundary[10 * width + cx] = 1;  // top edge
    boundary[(height - 10) * width + cx] = 1;  // bottom edge
    
    printf("Generated test pattern with 9 boundary points in %dx%d image\n", width, height);
    
    // Print boundary point locations for verification
    printf("Boundary points at: ");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (boundary[y * width + x] == 1) {
                printf("(%d,%d) ", x, y);
            }
        }
    }
    printf("\n");
}

int main() {
    printf("=== PBA-based EDT API Test (2D) ===\n");
    
    // Test parameters  
    const int width = 256;
    const int height = 256;
    const int num_boundary_points = 100;
    
    size_t image_size = width * height;
    size_t result_size = image_size * 2; // 2 components for 2D indices
    
    // Allocate host memory
    char* h_boundary = (char*)malloc(image_size * sizeof(char));
    int* h_pba_index = (int*)malloc(result_size * sizeof(int));
    float* h_pba_distance = (float*)malloc(image_size * sizeof(float));
    int* h_edt_index = (int*)malloc(result_size * sizeof(int));
    float* h_edt_distance = (float*)malloc(image_size * sizeof(float));
    
    if (!h_boundary || !h_pba_index || !h_pba_distance || !h_edt_index || !h_edt_distance) {
        printf("Error: Failed to allocate host memory\n");
        return -1;
    }
    
    // Generate test data
    printf("\n1. Generating test data...\n");
    generate_test_pattern(h_boundary, width, height);
    
    // Allocate device memory
    char* d_boundary;
    int* d_pba_index;
    float* d_pba_distance;
    int* d_edt_index;
    float* d_edt_distance;
    
    cudaMalloc(&d_boundary, image_size * sizeof(char));
    cudaMalloc(&d_pba_index, result_size * sizeof(int));
    cudaMalloc(&d_pba_distance, image_size * sizeof(float));
    cudaMalloc(&d_edt_index, result_size * sizeof(int));
    cudaMalloc(&d_edt_distance, image_size * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_boundary, h_boundary, image_size * sizeof(char), cudaMemcpyHostToDevice);
    
    // Test PBA-based EDT
    printf("\n2. Running PBA-based EDT...\n");
    pba_edt_2d(d_boundary, d_pba_index, d_pba_distance, width, height);
    
    // Test standard EDT for comparison
    printf("\n3. Running standard EDT for comparison...\n");
    edt_2d(d_boundary, d_edt_index, d_edt_distance, width, height);
    
    // Copy results back to host
    cudaMemcpy(h_pba_index, d_pba_index, result_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pba_distance, d_pba_distance, image_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_edt_index, d_edt_index, result_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_edt_distance, d_edt_distance, image_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify results
    printf("\n4. Verifying results...\n");
    int differences = 0;
    float max_distance_diff = 0.0f;
    float avg_distance_diff = 0.0f;
    
    for (int i = 0; i < image_size; i++) {
        float pba_dist = h_pba_distance[i];
        float edt_dist = h_edt_distance[i];
        float diff = fabsf(pba_dist - edt_dist);
        
        if (diff > 0.01f) { // Allow small floating-point differences
            differences++;
        }
        
        max_distance_diff = fmaxf(max_distance_diff, diff);
        avg_distance_diff += diff;
    }
    avg_distance_diff /= image_size;
    
    printf("Results comparison:\n");
    printf("- Pixels with significant differences: %d / %zu (%.2f%%)\n", 
           differences, image_size, 100.0f * differences / image_size);
    printf("- Maximum distance difference: %.6f\n", max_distance_diff);
    printf("- Average distance difference: %.6f\n", avg_distance_diff);
    
    // Sample output verification
    printf("\n5. Sample results (center region):\n");
    printf("Position\tPBA Distance\tEDT Distance\tPBA Index\tEDT Index\n");
    for (int y = height/2 - 2; y <= height/2 + 2; y++) {
        for (int x = width/2 - 2; x <= width/2 + 2; x++) {
            int idx = y * width + x;
            printf("(%3d,%3d)\t%.3f\t\t%.3f\t\t(%d,%d)\t\t(%d,%d)\n",
                   x, y,
                   h_pba_distance[idx], h_edt_distance[idx],
                   h_pba_index[idx*2+1], h_pba_index[idx*2+0],  // x, y
                   h_edt_index[idx*2+1], h_edt_index[idx*2+0]); // x, y
        }
    }
    
    // Save results to files
    printf("\n6. Saving results to files...\n");
    write_binary_file("pba_distances_2d.dat", h_pba_distance, image_size);
    write_binary_file("pba_indices_2d.dat", h_pba_index, result_size);
    write_binary_file("boundary_mask_2d.dat", h_boundary, image_size);
    
    // Performance benchmark
    printf("\n7. Running performance benchmark...\n");
    benchmark_pba_vs_edt_2d(d_boundary, width, height, 10);
    
    // Test with random boundary points
    printf("\n8. Testing with random boundary points...\n");
    generate_random_boundary(h_boundary, width, height, num_boundary_points);
    cudaMemcpy(d_boundary, h_boundary, image_size * sizeof(char), cudaMemcpyHostToDevice);
    
    pba_edt_2d(d_boundary, d_pba_index, d_pba_distance, width, height);
    cudaMemcpy(h_pba_distance, d_pba_distance, image_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Find some statistics
    float min_dist = h_pba_distance[0];
    float max_dist = h_pba_distance[0];
    float avg_dist = 0.0f;
    
    for (int i = 0; i < image_size; i++) {
        float dist = h_pba_distance[i];
        min_dist = fminf(min_dist, dist);
        max_dist = fmaxf(max_dist, dist);
        avg_dist += dist;
    }
    avg_dist /= image_size;
    
    printf("Distance statistics for random boundary:\n");
    printf("- Min distance: %.3f\n", min_dist);
    printf("- Max distance: %.3f\n", max_dist);
    printf("- Average distance: %.3f\n", avg_dist);
    
    // Test optimized version with power-of-2 size
    if (width == height && (width & (width - 1)) == 0) {
        printf("\n9. Testing optimized version...\n");
        pba_edt_2d_optimized(d_boundary, d_pba_index, d_pba_distance, width);
        printf("Optimized version completed successfully\n");
    } else {
        printf("\n9. Skipping optimized version (requires square power-of-2 dimensions)\n");
    }
    
    // Clean up
    free(h_boundary);
    free(h_pba_index);
    free(h_pba_distance);
    free(h_edt_index);
    free(h_edt_distance);
    
    cudaFree(d_boundary);
    cudaFree(d_pba_index);
    cudaFree(d_pba_distance);
    cudaFree(d_edt_index);
    cudaFree(d_edt_distance);
    
    printf("\n=== Test completed successfully! ===\n");
    return 0;
}
