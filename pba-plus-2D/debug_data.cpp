#include <stdio.h>
#include <stdlib.h>

int main() {
    // Read and analyze boundary mask
    FILE* f = fopen("boundary_mask_2d.dat", "rb");
    if (!f) {
        printf("Cannot open boundary_mask_2d.dat\n");
        return 1;
    }
    
    const int width = 256, height = 256;
    char* boundary = (char*)malloc(width * height);
    size_t read_size = fread(boundary, sizeof(char), width * height, f);
    fclose(f);
    
    printf("Read %zu bytes from boundary mask\n", read_size);
    
    int boundary_count = 0;
    printf("Boundary points found:\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            if (boundary[idx] == 1) {
                printf("  Point %d: (%d, %d)\n", boundary_count + 1, x, y);
                boundary_count++;
            }
        }
    }
    printf("Total boundary points: %d\n", boundary_count);
    
    // Read and analyze distance data
    f = fopen("pba_distances_2d.dat", "rb");
    if (!f) {
        printf("Cannot open pba_distances_2d.dat\n");
        free(boundary);
        return 1;
    }
    
    float* distances = (float*)malloc(width * height * sizeof(float));
    read_size = fread(distances, sizeof(float), width * height, f);
    fclose(f);
    
    printf("\nRead %zu floats from distance data\n", read_size);
    
    // Analyze distances
    float min_dist = distances[0], max_dist = distances[0];
    int zero_count = 0, non_zero_count = 0;
    
    for (int i = 0; i < width * height; i++) {
        if (distances[i] == 0.0f) {
            zero_count++;
        } else {
            non_zero_count++;
        }
        if (distances[i] < min_dist) min_dist = distances[i];
        if (distances[i] > max_dist) max_dist = distances[i];
    }
    
    printf("Distance statistics:\n");
    printf("  Min: %.6f, Max: %.6f\n", min_dist, max_dist);
    printf("  Zero distances: %d\n", zero_count);
    printf("  Non-zero distances: %d\n", non_zero_count);
    
    // Check some specific points
    printf("\nSample distances:\n");
    int sample_points[][2] = {{0,0}, {128,128}, {100,100}, {200,200}, {50,50}};
    for (int i = 0; i < 5; i++) {
        int x = sample_points[i][0];
        int y = sample_points[i][1];
        int idx = y * width + x;
        printf("  Point (%d,%d): boundary=%d, distance=%.6f\n", 
               x, y, boundary[idx], distances[idx]);
    }
    
    free(boundary);
    free(distances);
    return 0;
}
