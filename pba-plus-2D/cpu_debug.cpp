#include <stdio.h>
#include <math.h>

int main() {
    // Test the distance calculation logic on CPU
    const int width = 256, height = 256;
    
    // Read boundary data
    FILE* f = fopen("boundary_mask_2d.dat", "rb");
    if (!f) {
        printf("Cannot open boundary file\n");
        return 1;
    }
    
    char* boundary = (char*)malloc(width * height);
    fread(boundary, sizeof(char), width * height, f);
    fclose(f);
    
    // Test our EDT logic on a few sample points
    int test_points[][2] = {{100, 100}, {50, 50}, {200, 200}, {128, 128}};
    
    for (int t = 0; t < 4; t++) {
        int x = test_points[t][0];
        int y = test_points[t][1];
        int idx = y * width + x;
        
        printf("\nTesting point (%d, %d):\n", x, y);
        printf("  Is boundary: %s\n", boundary[idx] == 1 ? "YES" : "NO");
        
        float min_dist = 1000000.0f;
        int nearest_x = -1, nearest_y = -1;
        bool found_boundary = false;
        
        // Replicate the exact kernel logic
        for (int by = 0; by < height; by++) {
            for (int bx = 0; bx < width; bx++) {
                int b_idx = by * width + bx;
                if (boundary[b_idx] == 1) {
                    found_boundary = true;
                    float dx = (float)(bx - x);
                    float dy = (float)(by - y);
                    float dist = sqrtf(dx * dx + dy * dy);
                    
                    if (t == 0 && (bx == 0 || bx == 128 || bx == 255)) {
                        printf("    Boundary at (%d, %d): dx=%.1f, dy=%.1f, dist=%.3f\n",
                               bx, by, dx, dy, dist);
                    }
                    
                    if (dist < min_dist) {
                        min_dist = dist;
                        nearest_x = bx;
                        nearest_y = by;
                        
                        if (t == 0) {
                            printf("    New minimum: dist=%.3f from (%d, %d)\n", 
                                   min_dist, nearest_x, nearest_y);
                        }
                    }
                }
            }
        }
        
        printf("  Found boundary: %s\n", found_boundary ? "YES" : "NO");
        printf("  Final result: dist=%.6f, nearest=(%d, %d)\n", 
               min_dist, nearest_x, nearest_y);
    }
    
    free(boundary);
    return 0;
}
