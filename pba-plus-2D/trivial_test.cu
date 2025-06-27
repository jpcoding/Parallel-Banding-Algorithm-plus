#include <stdio.h>
#include <cuda_runtime.h>

__global__ void trivial_kernel(float* distance, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    distance[idx] = 42.0f;  // Just set all to 42 to verify kernel runs
}

int main() {
    const int width = 256, height = 256;
    const int size = width * height;
    
    // Allocate device memory
    float* d_distance;
    float* h_distance = (float*)malloc(size * sizeof(float));
    
    cudaMalloc(&d_distance, size * sizeof(float));
    
    // Initialize with zeros
    cudaMemset(d_distance, 0, size * sizeof(float));
    
    // Run kernel
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    trivial_kernel<<<grid, block>>>(d_distance, width, height);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Copy back
    cudaMemcpy(h_distance, d_distance, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Check results
    int correct_count = 0;
    for (int i = 0; i < size; i++) {
        if (h_distance[i] == 42.0f) {
            correct_count++;
        }
    }
    
    printf("Trivial kernel test:\n");
    printf("  Total pixels: %d\n", size);
    printf("  Pixels set to 42: %d\n", correct_count);
    printf("  Success: %s\n", (correct_count == size) ? "YES" : "NO");
    
    // Check some specific values
    printf("Sample values: %.1f %.1f %.1f %.1f\n", 
           h_distance[0], h_distance[1000], h_distance[50000], h_distance[size-1]);
    
    // Cleanup
    free(h_distance);
    cudaFree(d_distance);
    
    return 0;
}
