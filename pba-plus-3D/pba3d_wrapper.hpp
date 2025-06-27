#ifndef PBA3D_WRAPPER_HPP
#define PBA3D_WRAPPER_HPP

#include <cmath>
#include <cstdlib>
#include <stdexcept>

#include "pba/pba3D.h"

namespace PBA {

template <typename T_distance = double, typename T_int = int>
class PBA3D {
    struct Distance_and_Index {
        std::unique_ptr<T_distance[]> distance;
        std::unique_ptr<T_int[]> coordinates;  // x, y, z coordinates stored sequentially
    };

public:
    PBA3D() = default;
    
    ~PBA3D() {
        if (is_initialized) {
            cleanup();
        }
    }

    // Initialize the PBA algorithm with given texture size
    // textureSize must be power of 2 and >= 32
    bool initialize(int textureSize) {
        if (is_initialized) {
            cleanup();
        }
        
        this->texture_size = textureSize;
        
        // Initialize CUDA context
        pba3DInitialization(textureSize);
        
        // Allocate host memory
        input_voronoi = static_cast<int*>(malloc(textureSize * textureSize * textureSize * sizeof(int)));
        output_voronoi = static_cast<int*>(malloc(textureSize * textureSize * textureSize * sizeof(int)));
        
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
            free(input_voronoi);
            free(output_voronoi);
            input_voronoi = nullptr;
            output_voronoi = nullptr;
            is_initialized = false;
        }
    }

    // Set phase band parameters (optional, defaults will be used if not called)
    void set_phase_bands(int phase1Band = 1, int phase2Band = 1, int phase3Band = 2) {
        this->phase1_band = phase1Band;
        this->phase2_band = phase2Band;
        this->phase3_band = phase3Band;
    }

    // Compute 3D Voronoi diagram from a list of seed points
    // Returns distance field and nearest coordinate field
    Distance_and_Index compute_voronoi_from_points(const std::vector<std::tuple<int, int, int>>& seed_points) {
        if (!is_initialized) {
            throw std::runtime_error("PBA3D not initialized. Call initialize() first.");
        }

        // Clear input array
        int total_size = texture_size * texture_size * texture_size;
        for (int i = 0; i < total_size; i++) {
            input_voronoi[i] = MARKER;
        }

        // Set seed points
        for (const auto& point : seed_points) {
            int x, y, z;
            std::tie(x, y, z) = point;
            
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

        // Convert results to distance and coordinate arrays
        return convert_to_distance_and_coordinates();
    }

    // Compute 3D Voronoi diagram from a binary mask
    // mask should be texture_size^3 elements, with non-zero values indicating seed points
    Distance_and_Index compute_voronoi_from_mask(const char* mask) {
        if (!is_initialized) {
            throw std::runtime_error("PBA3D not initialized. Call initialize() first.");
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
    std::tuple<T_distance, int, int, int> query_point(double x, double y, double z) {
        if (!is_initialized) {
            throw std::runtime_error("PBA3D not initialized. Call initialize() first.");
        }

        // Convert to grid coordinates
        int grid_x = static_cast<int>(round(x));
        int grid_y = static_cast<int>(round(y));
        int grid_z = static_cast<int>(round(z));

        // Check bounds
        if (grid_x < 0 || grid_x >= texture_size || 
            grid_y < 0 || grid_y >= texture_size || 
            grid_z < 0 || grid_z >= texture_size) {
            return std::make_tuple(static_cast<T_distance>(-1), -1, -1, -1);
        }

        // Get nearest site
        int id = grid_z * texture_size * texture_size + grid_y * texture_size + grid_x;
        int nearest_x, nearest_y, nearest_z;
        DECODE(output_voronoi[id], nearest_x, nearest_y, nearest_z);

        // Calculate distance
        double dx = nearest_x - x;
        double dy = nearest_y - y;
        double dz = nearest_z - z;
        T_distance distance = static_cast<T_distance>(sqrt(dx * dx + dy * dy + dz * dz));

        return std::make_tuple(distance, nearest_x, nearest_y, nearest_z);
    }

    // Get texture size
    int get_texture_size() const { return texture_size; }

    // Check if initialized
    bool is_ready() const { return is_initialized; }

private:
    bool is_initialized = false;
    int texture_size = 0;
    int* input_voronoi = nullptr;
    int* output_voronoi = nullptr;
    
    // Phase band parameters
    int phase1_band = 1;
    int phase2_band = 1;
    int phase3_band = 2;

    Distance_and_Index convert_to_distance_and_coordinates() {
        int total_size = texture_size * texture_size * texture_size;
        
        auto distance = std::unique_ptr<T_distance[]>(
            static_cast<T_distance*>(malloc(total_size * sizeof(T_distance))));
        auto coordinates = std::unique_ptr<T_int[]>(
            static_cast<T_int*>(malloc(total_size * 3 * sizeof(T_int))));

        if (!distance || !coordinates) {
            throw std::runtime_error("Failed to allocate memory for results");
        }

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
                    distance[idx] = static_cast<T_distance>(sqrt(dx * dx + dy * dy + dz * dz));
                    
                    // Store coordinates
                    coordinates[idx * 3 + 0] = static_cast<T_int>(nearest_x);
                    coordinates[idx * 3 + 1] = static_cast<T_int>(nearest_y);
                    coordinates[idx * 3 + 2] = static_cast<T_int>(nearest_z);
                }
            }
        }

        return {std::move(distance), std::move(coordinates)};
    }
};

// Convenience function for simple usage
template <typename T_distance = double, typename T_int = int>
std::tuple<std::vector<T_distance>, std::vector<T_int>> 
compute_3d_voronoi_simple(const std::vector<std::tuple<int, int, int>>& seed_points, 
                         int texture_size) {
    PBA3D<T_distance, T_int> pba;
    
    if (!pba.initialize(texture_size)) {
        throw std::runtime_error("Failed to initialize PBA3D");
    }
    
    auto result = pba.compute_voronoi_from_points(seed_points);
    
    // Convert to vectors for easier usage
    int total_size = texture_size * texture_size * texture_size;
    std::vector<T_distance> distance_vec(total_size);
    std::vector<T_int> coordinates_vec(total_size * 3);
    
    std::copy(result.distance.get(), result.distance.get() + total_size, distance_vec.begin());
    std::copy(result.coordinates.get(), result.coordinates.get() + total_size * 3, coordinates_vec.begin());
    
    return std::make_tuple(std::move(distance_vec), std::move(coordinates_vec));
}

} // namespace PBA

#endif // PBA3D_WRAPPER_HPP
