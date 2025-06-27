#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void debug_boundary_kernel(char* boundary, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    char val = boundary[idx];
    
    // Print first few values
    if (idx < 16) {
        printf("GPU: boundary[%d] = %d (at %d,%d)\n", idx, (int)val, x, y);
    }
    
    // Print the specific coordinates we expect to have boundary points
    if ((x == 0 && y == 0) || (x == 3 && y == 3)) {
        printf("GPU: Expected boundary at (%d,%d) has value %d\n", x, y, (int)val);
    }
}

int main() {
    printf("=== Debug Boundary Transfer ===\n");
    
    const int width = 4;
    const int height = 4;
    size_t image_size = width * height;
    
    // Host array
    char* h_boundary = (char*)malloc(image_size);
    
    // Create boundary pattern
    memset(h_boundary, 0, image_size);
    h_boundary[0] = 1;  // (0,0)
    h_boundary[15] = 1; // (3,3)
    
    printf("Host boundary pattern:\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            printf("h_boundary[%d] = %d (at %d,%d)\n", idx, (int)h_boundary[idx], x, y);
        }
    }
    
    // Device array
    char* d_boundary;
    cudaMalloc(&d_boundary, image_size);
    
    // Copy to device
    cudaMemcpy(d_boundary, h_boundary, image_size, cudaMemcpyHostToDevice);
    
    // Launch debug kernel
    dim3 block(2, 2);
    dim3 grid(2, 2);
    debug_boundary_kernel<<<grid, block>>>(d_boundary, width, height);
    cudaDeviceSynchronize();
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    // Copy back and verify
    char* h_boundary_check = (char*)malloc(image_size);
    cudaMemcpy(h_boundary_check, d_boundary, image_size, cudaMemcpyDeviceToHost);
    
    printf("\nHost verification after device copy:\n");
    for (int i = 0; i < image_size; i++) {
        if (h_boundary[i] != h_boundary_check[i]) {
            printf("MISMATCH at index %d: original=%d, copied_back=%d\n", 
                   i, (int)h_boundary[i], (int)h_boundary_check[i]);
        }
    }
    
    free(h_boundary);
    free(h_boundary_check);
    cudaFree(d_boundary);
    
    return 0;
}
