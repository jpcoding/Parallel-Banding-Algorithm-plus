#include <stdio.h>
#include <cuda_runtime.h>

__global__ void simple_debug_kernel(char* boundary, float* distance, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Debug: let's see what values we're actually getting
    char boundary_val = boundary[idx];
    
    // Store the actual boundary value as the distance (to see what the GPU is reading)
    distance[idx] = (float)boundary_val;
}

int main() {
    const int width = 256, height = 256;
    const int size = width * height;
    
    // Read boundary data
    FILE* f = fopen("boundary_mask_2d.dat", "rb");
    if (!f) {
        printf("Cannot open boundary file\n");
        return 1;
    }
    
    char* h_boundary = (char*)malloc(size);
    fread(h_boundary, sizeof(char), size, f);
    fclose(f);
    
    // Count boundary points on host
    int boundary_count = 0;
    for (int i = 0; i < size; i++) {
        if (h_boundary[i] == 1) boundary_count++;
    }
    printf("Host counted %d boundary points\n", boundary_count);
    
    // Check a few specific locations on host
    printf("Host boundary values at specific locations:\n");
    int test_locs[][2] = {{0,0}, {128,128}, {100,100}, {255,255}};
    for (int i = 0; i < 4; i++) {
        int x = test_locs[i][0], y = test_locs[i][1];
        int idx = y * width + x;
        printf("  (%d,%d): value=%d\n", x, y, (int)h_boundary[idx]);
    }
    
    // Allocate device memory
    char* d_boundary;
    char* h_boundary_verify = (char*)malloc(size);
    
    cudaError_t err = cudaMalloc(&d_boundary, size);
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Copy to device
    err = cudaMemcpy(d_boundary, h_boundary, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Copy back immediately to verify transfer
    err = cudaMemcpy(h_boundary_verify, d_boundary, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Compare original and verified data
    int differences = 0;
    int verify_boundary_count = 0;
    for (int i = 0; i < size; i++) {
        if (h_boundary_verify[i] == 1) verify_boundary_count++;
        if (h_boundary[i] != h_boundary_verify[i]) {
            differences++;
            if (differences <= 10) {  // Show first 10 differences
                printf("Difference at index %d: original=%d, verified=%d\n", 
                       i, (int)h_boundary[i], (int)h_boundary_verify[i]);
            }
        }
    }
    
    printf("Memory transfer verification:\n");
    printf("  Original boundary count: %d\n", boundary_count);
    printf("  Verified boundary count: %d\n", verify_boundary_count);
    printf("  Differences: %d\n", differences);
    
    printf("Verified boundary values at specific locations:\n");
    for (int i = 0; i < 4; i++) {
        int x = test_locs[i][0], y = test_locs[i][1];
        int idx = y * width + x;
        printf("  (%d,%d): original=%d, verified=%d\n", 
               x, y, (int)h_boundary[idx], (int)h_boundary_verify[idx]);
    }
    
    // Cleanup
    free(h_boundary);
    free(h_boundary_verify);
    cudaFree(d_boundary);
    
    return 0;
}
