#ifndef PBA2D_WRAPPER_HPP
#define PBA2D_WRAPPER_HPP

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "pba/pba2D.h"

namespace PBA {

struct VoronoiResult2D {
    double* distances;     // Array of distances [size^2]
    int* coordinates;      // Array of coordinates [size^2 * 2] (x,y pairs)
    int size;              // Texture size
    int total_elements;    // size^2
};

struct QueryResult2D {
    double distance;
    int nearest_x, nearest_y;
    bool valid;
};

class PBA2D {
public:
    PBA2D() : is_initialized(false), texture_size(0), 
              input_voronoi(NULL), output_voronoi(NULL),
              phase1_band(1), phase2_band(1), phase3_band(2) {}
    
    ~PBA2D() {
        cleanup();
    }

    // Initialize the PBA algorithm with given texture size
    bool initialize(int textureSize) {
        if (is_initialized) {
            cleanup();
        }
        
        this->texture_size = textureSize;
        
        // Initialize CUDA context
        pba2DInitialization(textureSize, phase1_band);
        
        // Allocate host memory
        int total_size = textureSize * textureSize;
        input_voronoi = (short*)malloc(total_size * 2 * sizeof(short));
        output_voronoi = (short*)malloc(total_size * 2 * sizeof(short));
        
        if (!input_voronoi || !output_voronoi) {
            printf("Error: Failed to allocate memory for Voronoi arrays\n");
            cleanup();
            return false;
        }
        
        // Initialize input array with MARKER values
        for (int i = 0; i < total_size * 2; i++) {
            input_voronoi[i] = MARKER;
        }
        
        is_initialized = true;
        printf("PBA2D initialized with texture size %d\n", textureSize);
        return true;
    }
    
    // Clean up resources
    void cleanup() {
        if (is_initialized) {
            pba2DDeinitialization();
            is_initialized = false;
        }
        
        if (input_voronoi) {
            free(input_voronoi);
            input_voronoi = NULL;
        }
        
        if (output_voronoi) {
            free(output_voronoi);
            output_voronoi = NULL;
        }
        
        texture_size = 0;
    }
    
    // Add a seed point at the given coordinates
    bool addSeed(int x, int y) {
        if (!is_initialized) {
            printf("Error: PBA2D not initialized\n");
            return false;
        }
        
        if (x < 0 || x >= texture_size || y < 0 || y >= texture_size) {
            printf("Error: Seed coordinates (%d, %d) out of bounds [0, %d)\n", 
                   x, y, texture_size);
            return false;
        }
        
        int idx = y * texture_size + x;
        input_voronoi[idx * 2] = (short)x;
        input_voronoi[idx * 2 + 1] = (short)y;
        
        return true;
    }
    
    // Add multiple seed points from arrays
    bool addSeeds(const int* x_coords, const int* y_coords, int num_seeds) {
        if (!is_initialized) {
            printf("Error: PBA2D not initialized\n");
            return false;
        }
        
        for (int i = 0; i < num_seeds; i++) {
            if (!addSeed(x_coords[i], y_coords[i])) {
                return false;
            }
        }
        
        return true;
    }
    
    // Add seeds from a binary mask (1 = seed, 0 = empty)
    bool addSeedsFromMask(const char* mask) {
        if (!is_initialized) {
            printf("Error: PBA2D not initialized\n");
            return false;
        }
        
        for (int y = 0; y < texture_size; y++) {
            for (int x = 0; x < texture_size; x++) {
                int idx = y * texture_size + x;
                if (mask[idx] == 1) {
                    addSeed(x, y);
                }
            }
        }
        
        return true;
    }
    
    // Clear all seeds
    void clearSeeds() {
        if (!is_initialized) return;
        
        int total_size = texture_size * texture_size;
        for (int i = 0; i < total_size * 2; i++) {
            input_voronoi[i] = MARKER;
        }
    }
    
    // Compute the Voronoi diagram
    bool computeVoronoi() {
        if (!is_initialized) {
            printf("Error: PBA2D not initialized\n");
            return false;
        }
        
        // Run the PBA algorithm
        pba2DVoronoiDiagram(input_voronoi, output_voronoi, 
                           phase1_band, phase2_band, phase3_band);
        
        return true;
    }
    
    // Get the complete result as arrays
    VoronoiResult2D getResult() {
        VoronoiResult2D result = {NULL, NULL, 0, 0};
        
        if (!is_initialized) {
            printf("Error: PBA2D not initialized\n");
            return result;
        }
        
        result.size = texture_size;
        result.total_elements = texture_size * texture_size;
        
        // Allocate result arrays
        result.distances = (double*)malloc(result.total_elements * sizeof(double));
        result.coordinates = (int*)malloc(result.total_elements * 2 * sizeof(int));
        
        if (!result.distances || !result.coordinates) {
            printf("Error: Failed to allocate result arrays\n");
            if (result.distances) free(result.distances);
            if (result.coordinates) free(result.coordinates);
            result.distances = NULL;
            result.coordinates = NULL;
            return result;
        }
        
        // Process output and calculate distances
        for (int y = 0; y < texture_size; y++) {
            for (int x = 0; x < texture_size; x++) {
                int idx = y * texture_size + x;
                
                short nearest_x = output_voronoi[idx * 2];
                short nearest_y = output_voronoi[idx * 2 + 1];
                
                // Store coordinates
                result.coordinates[idx * 2] = nearest_x;
                result.coordinates[idx * 2 + 1] = nearest_y;
                
                // Calculate distance
                double dx = nearest_x - x;
                double dy = nearest_y - y;
                result.distances[idx] = sqrt(dx * dx + dy * dy);
            }
        }
        
        return result;
    }
    
    // Query a specific point
    QueryResult2D query(int x, int y) {
        QueryResult2D result = {0.0, 0, 0, false};
        
        if (!is_initialized) {
            printf("Error: PBA2D not initialized\n");
            return result;
        }
        
        if (x < 0 || x >= texture_size || y < 0 || y >= texture_size) {
            printf("Error: Query coordinates (%d, %d) out of bounds\n", x, y);
            return result;
        }
        
        int idx = y * texture_size + x;
        result.nearest_x = output_voronoi[idx * 2];
        result.nearest_y = output_voronoi[idx * 2 + 1];
        
        double dx = result.nearest_x - x;
        double dy = result.nearest_y - y;
        result.distance = sqrt(dx * dx + dy * dy);
        result.valid = true;
        
        return result;
    }
    
    // Get raw output array (for advanced users)
    const short* getRawOutput() const {
        return output_voronoi;
    }
    
    // Set band parameters (advanced)
    void setBandParameters(int phase1, int phase2, int phase3) {
        phase1_band = phase1;
        phase2_band = phase2;
        phase3_band = phase3;
    }
    
    // Get current parameters
    void getParameters(int& size, int& p1, int& p2, int& p3) const {
        size = texture_size;
        p1 = phase1_band;
        p2 = phase2_band;
        p3 = phase3_band;
    }
    
    // Check if initialized
    bool isInitialized() const {
        return is_initialized;
    }

private:
    bool is_initialized;
    int texture_size;
    short* input_voronoi;
    short* output_voronoi;
    int phase1_band, phase2_band, phase3_band;
};

// Helper function to free VoronoiResult2D
inline void freeVoronoiResult(VoronoiResult2D& result) {
    if (result.distances) {
        free(result.distances);
        result.distances = NULL;
    }
    if (result.coordinates) {
        free(result.coordinates);
        result.coordinates = NULL;
    }
    result.size = 0;
    result.total_elements = 0;
}

} // namespace PBA

#endif // PBA2D_WRAPPER_HPP
