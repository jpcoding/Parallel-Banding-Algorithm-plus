#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "pba2d_wrapper_clean.hpp"

using namespace PBA;

void print_separator(const char* title) {
    printf("\n");
    for (int i = 0; i < 60; i++) printf("=");
    printf("\n%s\n", title);
    for (int i = 0; i < 60; i++) printf("=");
    printf("\n");
}

void demonstrate_basic_usage() {
    print_separator("Basic PBA2D Usage Example");
    
    const int texture_size = 128;  // Must be power of 2
    
    // Create PBA2D instance
    PBA2D pba;
    
    // Initialize
    if (!pba.initialize(texture_size)) {
        printf("Failed to initialize PBA2D\n");
        return;
    }
    
    // Add some seed points
    printf("Adding seed points...\n");
    pba.addSeed(32, 32);    // Top-left region
    pba.addSeed(96, 32);    // Top-right region
    pba.addSeed(32, 96);    // Bottom-left region
    pba.addSeed(96, 96);    // Bottom-right region
    pba.addSeed(64, 64);    // Center
    
    // Compute Voronoi diagram
    printf("Computing Voronoi diagram...\n");
    if (!pba.computeVoronoi()) {
        printf("Failed to compute Voronoi diagram\n");
        return;
    }
    
    // Query some specific points
    printf("\nQuerying specific points:\n");
    int query_points[][2] = {
        {20, 20}, {50, 50}, {80, 80}, {100, 100}, {10, 100}
    };
    
    for (int i = 0; i < 5; i++) {
        int x = query_points[i][0];
        int y = query_points[i][1];
        QueryResult2D result = pba.query(x, y);
        
        if (result.valid) {
            printf("Point (%3d, %3d) -> Nearest seed: (%3d, %3d), Distance: %.2f\n",
                   x, y, result.nearest_x, result.nearest_y, result.distance);
        }
    }
    
    // Get complete result
    printf("\nGetting complete results...\n");
    VoronoiResult2D full_result = pba.getResult();
    
    if (full_result.distances && full_result.coordinates) {
        printf("Successfully computed %d distances\n", full_result.total_elements);
        
        // Find some statistics
        double min_dist = full_result.distances[0];
        double max_dist = full_result.distances[0];
        double avg_dist = 0.0;
        
        for (int i = 0; i < full_result.total_elements; i++) {
            double dist = full_result.distances[i];
            if (dist < min_dist) min_dist = dist;
            if (dist > max_dist) max_dist = dist;
            avg_dist += dist;
        }
        avg_dist /= full_result.total_elements;
        
        printf("Distance statistics:\n");
        printf("  Min: %.2f, Max: %.2f, Avg: %.2f\n", min_dist, max_dist, avg_dist);
        
        // Clean up result
        freeVoronoiResult(full_result);
    }
    
    printf("\nBasic usage example completed!\n");
}

void demonstrate_mask_input() {
    print_separator("Mask Input Example");
    
    const int texture_size = 64;
    
    PBA2D pba;
    if (!pba.initialize(texture_size)) {
        printf("Failed to initialize PBA2D\n");
        return;
    }
    
    // Create a binary mask with a cross pattern
    char* mask = (char*)malloc(texture_size * texture_size);
    if (!mask) {
        printf("Failed to allocate mask memory\n");
        return;
    }
    
    // Initialize mask to 0
    for (int i = 0; i < texture_size * texture_size; i++) {
        mask[i] = 0;
    }
    
    // Create cross pattern
    int center = texture_size / 2;
    for (int i = 0; i < texture_size; i++) {
        // Horizontal line
        mask[center * texture_size + i] = 1;
        // Vertical line
        mask[i * texture_size + center] = 1;
    }
    
    printf("Created cross pattern mask\n");
    
    // Add seeds from mask
    pba.addSeedsFromMask(mask);
    
    // Compute
    if (pba.computeVoronoi()) {
        // Test some points
        printf("\nTesting points near the cross:\n");
        int test_points[][2] = {
            {center-10, center-10},  // Diagonal from center
            {center+10, center+10},  // Diagonal from center
            {center, center-5},      // On vertical line
            {center-5, center},      // On horizontal line
            {10, 10}                 // Corner
        };
        
        for (int i = 0; i < 5; i++) {
            QueryResult2D result = pba.query(test_points[i][0], test_points[i][1]);
            if (result.valid) {
                printf("Point (%2d, %2d) -> Nearest: (%2d, %2d), Distance: %.2f\n",
                       test_points[i][0], test_points[i][1],
                       result.nearest_x, result.nearest_y, result.distance);
            }
        }
    }
    
    free(mask);
    printf("\nMask input example completed!\n");
}

void demonstrate_performance() {
    print_separator("Performance Demonstration");
    
    const int sizes[] = {64, 128, 256, 512};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("Testing performance with different texture sizes:\n");
    printf("Size\tSeeds\tTime (ms)\tPixels/sec\n");
    printf("----\t-----\t---------\t----------\n");
    
    for (int s = 0; s < num_sizes; s++) {
        int size = sizes[s];
        
        PBA2D pba;
        if (!pba.initialize(size)) {
            printf("Failed to initialize size %d\n", size);
            continue;
        }
        
        // Add random seeds
        srand(42);  // Fixed seed for reproducibility
        int num_seeds = size * size / 100;  // 1% of pixels as seeds
        
        for (int i = 0; i < num_seeds; i++) {
            int x = rand() % size;
            int y = rand() % size;
            pba.addSeed(x, y);
        }
        
        // Time the computation
        clock_t start = clock();
        bool success = pba.computeVoronoi();
        clock_t end = clock();
        
        if (success) {
            double time_ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;
            int total_pixels = size * size;
            double pixels_per_sec = total_pixels / (time_ms / 1000.0);
            
            printf("%d\t%d\t%.2f\t\t%.0f\n", size, num_seeds, time_ms, pixels_per_sec);
        } else {
            printf("%d\t%d\tFAILED\t\t-\n", size, num_seeds);
        }
    }
    
    printf("\nPerformance demonstration completed!\n");
}

void demonstrate_advanced_features() {
    print_separator("Advanced Features");
    
    const int texture_size = 128;
    
    PBA2D pba;
    if (!pba.initialize(texture_size)) {
        printf("Failed to initialize PBA2D\n");
        return;
    }
    
    // Test different band parameters
    printf("Testing different band parameters:\n");
    
    // Add a grid of seeds
    for (int y = 20; y < texture_size; y += 40) {
        for (int x = 20; x < texture_size; x += 40) {
            pba.addSeed(x, y);
        }
    }
    
    int band_configs[][3] = {
        {1, 1, 2},  // Default
        {2, 2, 4},  // Larger bands
        {1, 2, 1},  // Different phase emphasis
    };
    
    for (int config = 0; config < 3; config++) {
        printf("\nTesting band configuration (%d, %d, %d):\n", 
               band_configs[config][0], band_configs[config][1], band_configs[config][2]);
        
        pba.setBandParameters(band_configs[config][0], 
                             band_configs[config][1], 
                             band_configs[config][2]);
        
        clock_t start = clock();
        bool success = pba.computeVoronoi();
        clock_t end = clock();
        
        if (success) {
            double time_ms = 1000.0 * (end - start) / CLOCKS_PER_SEC;
            printf("  Computation time: %.2f ms\n", time_ms);
            
            // Test a point
            QueryResult2D result = pba.query(50, 50);
            if (result.valid) {
                printf("  Point (50, 50) -> Nearest: (%d, %d), Distance: %.2f\n",
                       result.nearest_x, result.nearest_y, result.distance);
            }
        } else {
            printf("  Computation FAILED\n");
        }
    }
    
    // Test parameter retrieval
    int size, p1, p2, p3;
    pba.getParameters(size, p1, p2, p3);
    printf("\nCurrent parameters: size=%d, phases=(%d,%d,%d)\n", size, p1, p2, p3);
    
    printf("\nAdvanced features demonstration completed!\n");
}

int main() {
    printf("PBA2D Wrapper Example Program\n");
    
    // Run all demonstrations
    demonstrate_basic_usage();
    demonstrate_mask_input();
    demonstrate_performance();
    demonstrate_advanced_features();
    
    print_separator("All Examples Completed Successfully!");
    
    return 0;
}
