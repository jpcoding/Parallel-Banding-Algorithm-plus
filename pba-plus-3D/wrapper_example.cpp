#include <stdio.h>
#include <stdlib.h>
#include "pba3d_wrapper_clean.hpp"

using namespace PBA;

int main() {
    // Example usage of the PBA3D wrapper
    printf("=== PBA3D Wrapper Example ===\n");
    
    // Set up parameters
    int texture_size = 256;
    int num_seeds = 10;
    
    // Create some random seed points
    int seed_x[10] = {10, 50, 100, 150, 200, 30, 80, 120, 180, 220};
    int seed_y[10] = {20, 60, 110, 160, 210, 40, 90, 130, 190, 230};
    int seed_z[10] = {15, 55, 105, 155, 205, 35, 85, 125, 185, 225};
    
    // Method 1: Using the simple convenience function
    printf("\\nMethod 1: Using convenience function\\n");
    VoronoiResult result1 = compute_3d_voronoi_simple(seed_x, seed_y, seed_z, num_seeds, texture_size);
    
    if (result1.distances && result1.coordinates) {
        printf("Success! Generated %dx%dx%d Voronoi diagram\\n", 
               result1.size, result1.size, result1.size);
        
        // Query a few points
        printf("\\nSample distances and nearest coordinates:\\n");
        for (int i = 0; i < 5; i++) {
            int idx = i * 1000;  // Sample every 1000th element
            if (idx < result1.total_elements) {
                double dist = result1.distances[idx];
                int nx = result1.coordinates[idx * 3 + 0];
                int ny = result1.coordinates[idx * 3 + 1]; 
                int nz = result1.coordinates[idx * 3 + 2];
                printf("  Point %d: distance=%.3f, nearest=(%d,%d,%d)\\n", 
                       idx, dist, nx, ny, nz);
            }
        }
        
        free_voronoi_result(&result1);
    } else {
        printf("Failed to compute Voronoi diagram\\n");
    }
    
    // Method 2: Using the class directly
    printf("\\nMethod 2: Using PBA3D class directly\\n");
    PBA3D pba;
    
    if (pba.initialize(texture_size)) {
        printf("PBA3D initialized successfully\\n");
        
        // Set custom phase bands (optional)
        pba.set_phase_bands(1, 1, 2);
        
        // Compute Voronoi diagram
        VoronoiResult result2 = pba.compute_voronoi_from_points(seed_x, seed_y, seed_z, num_seeds);
        
        if (result2.distances && result2.coordinates) {
            printf("Voronoi computation successful\\n");
            
            // Test point queries
            printf("\\nPoint query examples:\\n");
            QueryResult q1 = pba.query_point(25.5, 35.7, 45.2);
            if (q1.valid) {
                printf("  Query (25.5, 35.7, 45.2): distance=%.3f, nearest=(%d,%d,%d)\\n",
                       q1.distance, q1.nearest_x, q1.nearest_y, q1.nearest_z);
            }
            
            QueryResult q2 = pba.query_point(100.0, 100.0, 100.0);
            if (q2.valid) {
                printf("  Query (100.0, 100.0, 100.0): distance=%.3f, nearest=(%d,%d,%d)\\n",
                       q2.distance, q2.nearest_x, q2.nearest_y, q2.nearest_z);
            }
            
            free_voronoi_result(&result2);
        } else {
            printf("Failed to compute Voronoi diagram\\n");
        }
    } else {
        printf("Failed to initialize PBA3D\\n");
    }
    
    printf("\\n=== Example completed ===\\n");
    return 0;
}
