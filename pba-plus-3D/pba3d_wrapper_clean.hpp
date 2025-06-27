#ifndef PBA3D_WRAPPER_HPP
#define PBA3D_WRAPPER_HPP

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "pba/pba3D.h"

namespace PBA {

struct VoronoiResult {
    double* distances;     // Array of distances [size^3]
    int* coordinates;      // Array of coordinates [size^3 * 3] (x,y,z triplets)
    int size;              // Texture size
    int total_elements;    // size^3
};

struct QueryResult {
    double distance;
    int nearest_x, nearest_y, nearest_z;
    bool valid;
};

class PBA3D {
public:
    PBA3D() : is_initialized(false), texture_size(0), 
              input_voronoi(NULL), output_voronoi(NULL),
              phase1_band(1), phase2_band(1), phase3_band(2) {}
    
    ~PBA3D() {
        cleanup();
    }

    // Initialize the PBA algorithm with given texture size
    bool initialize(int textureSize) {
        if (is_initialized) {
            cleanup();
        }
        
        this->texture_size = textureSize;
        
        // Initialize CUDA context
        pba3DInitialization(textureSize);
        
        // Allocate host memory
        int total_size = textureSize * textureSize * textureSize;
        input_voronoi = (int*)malloc(total_size * sizeof(int));
        output_voronoi = (int*)malloc(total_size * sizeof(int));
        
        if (!input_voronoi || !output_voronoi) {
            cleanup();
            return false;
        }
        
        is_initialized = true;
        return true;
    }

    // Clean up resources
    void cleanup() {
        if (is_initialized) {
            pba3DDeinitialization();
            if (input_voronoi) {
                free(input_voronoi);
                input_voronoi = NULL;
            }
            if (output_voronoi) {
                free(output_voronoi);
                output_voronoi = NULL;
            }
            is_initialized = false;
        }
    }

    // Set phase band parameters
    void set_phase_bands(int phase1Band = 1, int phase2Band = 1, int phase3Band = 2) {
        this->phase1_band = phase1Band;
        this->phase2_band = phase2Band;
        this->phase3_band = phase3Band;
    }

    // Compute 3D Voronoi diagram from seed point arrays
    VoronoiResult compute_voronoi_from_points(const int* seed_x, const int* seed_y, const int* seed_z, int num_seeds) {
        VoronoiResult result = {NULL, NULL, 0, 0};
        
        if (!is_initialized) {
            printf("Error: PBA3D not initialized\n");
            return result;
        }

        // Clear input array
        int total_size = texture_size * texture_size * texture_size;
        for (int i = 0; i < total_size; i++) {
            input_voronoi[i] = MARKER;
        }

        // Set seed points
        for (int i = 0; i < num_seeds; i++) {
            int x = seed_x[i];
            int y = seed_y[i]; 
            int z = seed_z[i];
            
            if (x >= 0 && x < texture_size && 
                y >= 0 && y < texture_size && 
                z >= 0 && z < texture_size) {
                int id = z * texture_size * texture_size + y * texture_size + x;
                input_voronoi[id] = ENCODE(x, y, z, 0, 0);
            }
        }

        // Run PBA algorithm
        pba3DVoronoiDiagram(input_voronoi, output_voronoi, 
                           phase1_band, phase2_band, phase3_band);

        // Convert results
        return convert_to_distance_and_coordinates();
    }

    // Compute 3D Voronoi diagram from a binary mask
    VoronoiResult compute_voronoi_from_mask(const char* mask) {
        VoronoiResult result = {NULL, NULL, 0, 0};
        
        if (!is_initialized) {
            printf("Error: PBA3D not initialized\n");
            return result;
        }

        // Clear input array and set from mask
        int total_size = texture_size * texture_size * texture_size;
        for (int i = 0; i < total_size; i++) {
            input_voronoi[i] = MARKER;
        }

        for (int z = 0; z < texture_size; z++) {
            for (int y = 0; y < texture_size; y++) {
                for (int x = 0; x < texture_size; x++) {
                    int idx = z * texture_size * texture_size + y * texture_size + x;
                    if (mask[idx] != 0) {
                        input_voronoi[idx] = ENCODE(x, y, z, 0, 0);
                    }
                }
            }
        }

        // Run PBA algorithm
        pba3DVoronoiDiagram(input_voronoi, output_voronoi, 
                           phase1_band, phase2_band, phase3_band);

        return convert_to_distance_and_coordinates();
    }

    // Query single point for nearest site and distance
    QueryResult query_point(double x, double y, double z) {
        QueryResult result = {-1.0, -1, -1, -1, false};
        
        if (!is_initialized) {
            return result;
        }

        // Convert to grid coordinates
        int grid_x = (int)(x + 0.5);  // Round to nearest integer
        int grid_y = (int)(y + 0.5);
        int grid_z = (int)(z + 0.5);

        // Check bounds
        if (grid_x < 0 || grid_x >= texture_size || 
            grid_y < 0 || grid_y >= texture_size || 
            grid_z < 0 || grid_z >= texture_size) {
            return result;
        }

        // Get nearest site
        int id = grid_z * texture_size * texture_size + grid_y * texture_size + grid_x;
        DECODE(output_voronoi[id], result.nearest_x, result.nearest_y, result.nearest_z);

        // Calculate distance
        double dx = result.nearest_x - x;
        double dy = result.nearest_y - y;
        double dz = result.nearest_z - z;
        result.distance = sqrt(dx * dx + dy * dy + dz * dz);
        result.valid = true;

        return result;
    }

    // Get texture size
    int get_texture_size() const { return texture_size; }

    // Check if initialized
    bool is_ready() const { return is_initialized; }

private:
    bool is_initialized;
    int texture_size;
    int* input_voronoi;
    int* output_voronoi;
    
    // Phase band parameters
    int phase1_band;
    int phase2_band;
    int phase3_band;

    VoronoiResult convert_to_distance_and_coordinates() {
        VoronoiResult result = {NULL, NULL, 0, 0};
        
        int total_size = texture_size * texture_size * texture_size;
        
        result.distances = (double*)malloc(total_size * sizeof(double));
        result.coordinates = (int*)malloc(total_size * 3 * sizeof(int));
        
        if (!result.distances || !result.coordinates) {
            if (result.distances) free(result.distances);
            if (result.coordinates) free(result.coordinates);
            printf("Error: Failed to allocate memory for results\n");
            return result;
        }

        result.size = texture_size;
        result.total_elements = total_size;

        for (int z = 0; z < texture_size; z++) {
            for (int y = 0; y < texture_size; y++) {
                for (int x = 0; x < texture_size; x++) {
                    int idx = z * texture_size * texture_size + y * texture_size + x;
                    
                    // Get nearest site coordinates
                    int nearest_x, nearest_y, nearest_z;
                    DECODE(output_voronoi[idx], nearest_x, nearest_y, nearest_z);
                    
                    // Calculate distance
                    int dx = nearest_x - x;
                    int dy = nearest_y - y;
                    int dz = nearest_z - z;
                    result.distances[idx] = sqrt((double)(dx * dx + dy * dy + dz * dz));
                    
                    // Store coordinates
                    result.coordinates[idx * 3 + 0] = nearest_x;
                    result.coordinates[idx * 3 + 1] = nearest_y;
                    result.coordinates[idx * 3 + 2] = nearest_z;
                }
            }
        }

        return result;
    }
};

// Helper function to free VoronoiResult
void free_voronoi_result(VoronoiResult* result) {
    if (result->distances) {
        free(result->distances);
        result->distances = NULL;
    }
    if (result->coordinates) {
        free(result->coordinates);
        result->coordinates = NULL;
    }
    result->size = 0;
    result->total_elements = 0;
}

// Convenience function for simple usage
VoronoiResult compute_3d_voronoi_simple(const int* seed_x, const int* seed_y, const int* seed_z, 
                                       int num_seeds, int texture_size) {
    PBA3D pba;
    VoronoiResult result = {NULL, NULL, 0, 0};
    
    if (!pba.initialize(texture_size)) {
        printf("Error: Failed to initialize PBA3D\n");
        return result;
    }
    
    return pba.compute_voronoi_from_points(seed_x, seed_y, seed_z, num_seeds);
}

} // namespace PBA

#endif // PBA3D_WRAPPER_HPP
