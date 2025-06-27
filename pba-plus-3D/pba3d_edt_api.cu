#include "pba3d_edt_api.hpp"
#include "pba/pba3D.h"
#include <algorithm>
#include <cstring>
#include <stdio.h>
#include <float.h>

// CUDA kernel to convert boundary mask to PBA input format
__global__ void init_pba_3d(char* boundary, int* pba_input, char b_tag, 
                            int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= width || y >= height || z >= depth) return;
    
    int idx = z * width * height + y * width + x;
    
    if (boundary[idx] == b_tag) {
        // This is a seed point - encode coordinates
        pba_input[idx] = ENCODE(x, y, z, 0, 0);
    } else {
        // Not a seed point - use marker
        pba_input[idx] = MARKER;
    }
}

// CUDA kernel to convert PBA output to index format and calculate distances
__global__ void pba_to_results(int* pba_output, int* index, float* distance,
                               int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= width || y >= height || z >= depth) return;
    
    int idx = z * width * height + y * width + x;
    
    // Decode nearest site coordinates from PBA output
    int nearest_x, nearest_y, nearest_z;
    DECODE(pba_output[idx], nearest_x, nearest_y, nearest_z);
    
    // Store in index array (z, y, x format to match your EDT)
    index[idx * 3 + 0] = nearest_z;
    index[idx * 3 + 1] = nearest_y; 
    index[idx * 3 + 2] = nearest_x;
    
    // Calculate distance
    float dx = nearest_x - x;
    float dy = nearest_y - y;
    float dz = nearest_z - z;
    distance[idx] = sqrtf(dx * dx + dy * dy + dz * dz);
}

// PBA-based EDT implementation with same API as your edt_3d
void pba_edt_3d(char* d_boundary, int* index, float* distance, 
                unsigned int width, unsigned int height, unsigned int depth) {
    
    size_t size = width * height * depth;
    
    // Allocate temporary arrays for PBA
    int* d_pba_input;
    int* d_pba_output;
    
    cudaMalloc(&d_pba_input, size * sizeof(int));
    cudaMalloc(&d_pba_output, size * sizeof(int));
    
    // Set up grid and block dimensions
    dim3 block(8, 8, 8);
    dim3 grid((width + block.x - 1) / block.x, 
              (height + block.y - 1) / block.y,
              (depth + block.z - 1) / block.z);
    
    // Convert boundary mask to PBA input format
    init_pba_3d<<<grid, block>>>(d_boundary, d_pba_input, (char)1, width, height, depth);
    cudaDeviceSynchronize();
    
    // Initialize PBA with the texture size (must be power of 2)
    int texture_size = std::max({width, height, depth});
    // Round up to next power of 2
    int pba_size = 32; // Minimum size
    while (pba_size < texture_size) {
        pba_size *= 2;
    }
    
    // If dimensions don't match PBA requirements, we need to pad
    if (width == height && height == depth && width == pba_size) {
        // Perfect match - use PBA directly
        pba3DInitialization(pba_size);
        pba3DVoronoiDiagram(d_pba_input, d_pba_output, 1, 1, 2);
        pba3DDeinitialization();
    } else {
        // Need to handle non-square or non-power-of-2 dimensions
        // For now, fall back to a simplified approach
        // You might want to implement padding/cropping here
        
        // Use the largest dimension as PBA size
        pba3DInitialization(pba_size);
        
        // Allocate padded arrays
        int* d_pba_padded_input;
        int* d_pba_padded_output;
        size_t padded_size = pba_size * pba_size * pba_size;
        
        cudaMalloc(&d_pba_padded_input, padded_size * sizeof(int));
        cudaMalloc(&d_pba_padded_output, padded_size * sizeof(int));
        
        // Initialize padded arrays with MARKER
        cudaMemset(d_pba_padded_input, 0xFF, padded_size * sizeof(int)); // MARKER = -1
        
        // Copy original data to padded array (simple copy for now)
        // This is a simplified version - you might want to implement proper 3D copying
        if (width <= pba_size && height <= pba_size && depth <= pba_size) {
            for (int z = 0; z < depth; z++) {
                for (int y = 0; y < height; y++) {
                    size_t src_offset = z * width * height + y * width;
                    size_t dst_offset = z * pba_size * pba_size + y * pba_size;
                    cudaMemcpy(d_pba_padded_input + dst_offset, 
                              d_pba_input + src_offset, 
                              width * sizeof(int), cudaMemcpyDeviceToDevice);
                }
            }
        }
        
        // Run PBA
        pba3DVoronoiDiagram(d_pba_padded_input, d_pba_padded_output, 1, 1, 2);
        
        // Copy results back
        for (int z = 0; z < depth; z++) {
            for (int y = 0; y < height; y++) {
                size_t src_offset = z * pba_size * pba_size + y * pba_size;
                size_t dst_offset = z * width * height + y * width;
                cudaMemcpy(d_pba_output + dst_offset,
                          d_pba_padded_output + src_offset,
                          width * sizeof(int), cudaMemcpyDeviceToDevice);
            }
        }
        
        cudaFree(d_pba_padded_input);
        cudaFree(d_pba_padded_output);
        pba3DDeinitialization();
    }
    
    // Convert PBA output to your format and calculate distances
    pba_to_results<<<grid, block>>>(d_pba_output, index, distance, width, height, depth);
    cudaDeviceSynchronize();
    
    // Clean up
    cudaFree(d_pba_input);
    cudaFree(d_pba_output);
}

// Optimized version that assumes cubic power-of-2 dimensions
void pba_edt_3d_optimized(char* d_boundary, int* index, float* distance, 
                          unsigned int size) {
    // This version assumes size is power of 2 and width=height=depth=size
    
    size_t total_elements = size * size * size;
    
    // Allocate temporary arrays for PBA
    int* d_pba_input;
    int* d_pba_output;
    
    cudaMalloc(&d_pba_input, total_elements * sizeof(int));
    cudaMalloc(&d_pba_output, total_elements * sizeof(int));
    
    // Set up grid and block dimensions
    dim3 block(8, 8, 8);
    dim3 grid((size + block.x - 1) / block.x, 
              (size + block.y - 1) / block.y,
              (size + block.z - 1) / block.z);
    
    // Convert boundary mask to PBA input format
    init_pba_3d<<<grid, block>>>(d_boundary, d_pba_input, (char)1, size, size, size);
    cudaDeviceSynchronize();
    
    // Run PBA
    pba3DInitialization(size);
    pba3DVoronoiDiagram(d_pba_input, d_pba_output, 1, 1, 2);
    pba3DDeinitialization();
    
    // Convert PBA output to your format and calculate distances
    pba_to_results<<<grid, block>>>(d_pba_output, index, distance, size, size, size);
    cudaDeviceSynchronize();
    
    // Clean up
    cudaFree(d_pba_input);
    cudaFree(d_pba_output);
}

// Proper CUDA EDT implementation for comparison
// width, height are planes, depth is the len, first dimension
__global__ void edt_depth(
    int* d_output, size_t stride, unsigned int rank, unsigned int d, unsigned int len, unsigned int width, unsigned int height,
    size_t width_stride, size_t height_stride)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;  // width
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;  // height
  if (x >= width || y >= height) return;

  int* d_output_start = d_output + x * width_stride + y * height_stride;
  int l = -1, ii, maxl, idx1, idx2, jj;
  int f[512][3];
  int g[512];

  int coor[3];
  coor[d] = 0;
  coor[(d + 1) % 3] = x;
  coor[(d + 2) % 3] = y;

  for (ii = 0; ii < len; ii++) {
    for (jj = 0; jj < rank; jj++) {
      f[ii][jj] = d_output_start[ii * stride + jj];
    }
  }

  for (ii = 0; ii < len; ii++) {
    if (f[ii][0] >= 0) {
      int fd = f[ii][d];
      int wR = 0.0;
      for (jj = 0; jj < rank; jj++) {
        if (jj != d) {
          int tw = (f[ii][jj] - coor[jj]);
          wR += tw * tw;
        }
      }
      while (l >= 1) {
        int a, b, c, uR = 0.0, vR = 0.0, f1;
        idx1 = g[l];
        f1 = f[idx1][d];
        idx2 = g[l - 1];
        a = f1 - f[idx2][d];
        b = fd - f1;
        c = a + b;
        for (jj = 0; jj < rank; jj++) {
          if (jj != d) {
            int cc = coor[jj];
            int tu = f[idx2][jj] - cc;
            int tv = f[idx1][jj] - cc;
            uR += tu * tu;
            vR += tv * tv;
          }
        }
        if (c * vR - b * uR - a * wR - a * b * c <= 0.0) { break; }
        --l;
      }
      ++l;
      g[l] = ii;
    }
  }
  maxl = l;
  if (maxl >= 0) {
    l = 0;
    for (ii = 0; ii < len; ii++) {
      int delta1 = 0.0, t;
      for (jj = 0; jj < rank; jj++) {
        t = jj == d ? f[g[l]][jj] - ii : f[g[l]][jj] - coor[jj];
        delta1 += t * t;
      }
      while (l < maxl) {
        int delta2 = 0.0;
        for (jj = 0; jj < rank; jj++) {
          t = jj == d ? f[g[l + 1]][jj] - ii : f[g[l + 1]][jj] - coor[jj];
          delta2 += t * t;
        }
        if (delta1 <= delta2) break;
        delta1 = delta2;
        ++l;
      }
      idx1 = g[l];
      for (jj = 0; jj < rank; jj++) d_output_start[ii * stride + jj] = f[idx1][jj];
    }
  }
}

__global__ void init_edt_3d(
    char* input, int* output, char b_tag, int rank, unsigned int width, unsigned int height, unsigned int depth)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;  // fastest dimension
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;  // slowest dimension
  if (x >= width || y >= height || z >= depth) return;
  int idx = x + y * width + z * width * height;

  if (input[idx] == b_tag) {
    output[idx * rank] = z;
    output[idx * rank + 1] = y;
    output[idx * rank + 2] = x;
  }
  else {
    output[idx * rank] = -1;
  }
}

__global__ void calculate_distance(
    int* output, float* distance, int rank, unsigned int width, unsigned int height, unsigned int depth)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;  // fastest dimension
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;  // depth
  if (x >= width || y >= height || z >= depth) return;
  int idx = x + y * width + z * width * height;
  double d = 0;
  d += (output[idx * rank] - z) * (output[idx * rank] - z);
  d += (output[idx * rank + 1] - y) * (output[idx * rank + 1] - y);
  d += (output[idx * rank + 2] - x) * (output[idx * rank + 2] - x);
  distance[idx] = sqrt(d);
}

// Proper CUDA EDT implementation
void edt_3d(char* d_boundary, int* index, float* distance, unsigned int width, unsigned int height, unsigned int depth)
{
  size_t width_stride = 3;
  size_t height_stride = width * 3;
  size_t depth_stride = width * height * 3;

  dim3 block(8, 8, 8);
  dim3 grid(
      (width + block.x - 1) / block.x, (height + block.y - 1) / block.y,
      (depth + block.z - 1) / block.z);
  dim3 block1(32, 32);
  dim3 grid1((height + block1.x - 1) / block1.x, (width + block1.y - 1) / block1.y);
  dim3 block3(32, 32);
  dim3 grid3((width + block3.x - 1) / block3.x, (depth + block3.y - 1) / block3.y);
  dim3 block2(32, 32);
  dim3 grid2((depth + block2.x - 1) / block2.x, (height + block2.y - 1) / block2.y);

  init_edt_3d<<<grid, block>>>(d_boundary, index, (char)1, (int)3, width, height, depth);
  cudaDeviceSynchronize();
  edt_depth<<<grid1, block1>>>(
      index, depth_stride, 3, 0, depth, height, width, height_stride, width_stride);
  cudaDeviceSynchronize();
  edt_depth<<<grid3, block3>>>(
      index, height_stride, 3, 1, height, width, depth, width_stride, depth_stride);
  cudaDeviceSynchronize();
  edt_depth<<<grid2, block2>>>(
      index, width_stride, 3, 2, width, depth, height, depth_stride, height_stride);
  cudaDeviceSynchronize();
  calculate_distance<<<grid, block>>>(index, distance, 3, width, height, depth);
  cudaDeviceSynchronize();
}

// Simple brute-force EDT implementation for comparison (only for very small volumes)
__global__ void brute_force_edt_kernel(char* boundary, int* index, float* distance,
                                       int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= width || y >= height || z >= depth) return;
    
    int idx = z * width * height + y * width + x;
    
    float min_dist = FLT_MAX;
    int nearest_x = -1, nearest_y = -1, nearest_z = -1;
    
    // Brute force: check distance to every boundary point
    for (int bz = 0; bz < depth; bz++) {
        for (int by = 0; by < height; by++) {
            for (int bx = 0; bx < width; bx++) {
                int b_idx = bz * width * height + by * width + bx;
                if (boundary[b_idx] == 1) {
                    float dx = bx - x;
                    float dy = by - y; 
                    float dz = bz - z;
                    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
                    
                    if (dist < min_dist) {
                        min_dist = dist;
                        nearest_x = bx;
                        nearest_y = by;
                        nearest_z = bz;
                    }
                }
            }
        }
    }
    
    distance[idx] = min_dist;
    index[idx * 3 + 0] = nearest_z;
    index[idx * 3 + 1] = nearest_y;
    index[idx * 3 + 2] = nearest_x;
}

// Simple brute-force EDT for comparison (only for very small volumes)
void brute_force_edt_3d(char* d_boundary, int* index, float* distance,
                        unsigned int width, unsigned int height, unsigned int depth) {
    dim3 block(8, 8, 8);
    dim3 grid((width + block.x - 1) / block.x, 
              (height + block.y - 1) / block.y,
              (depth + block.z - 1) / block.z);
    
    brute_force_edt_kernel<<<grid, block>>>(d_boundary, index, distance, width, height, depth);
    cudaDeviceSynchronize();
}

// Benchmark function to compare PBA vs proper EDT
void benchmark_pba_vs_edt(char* d_boundary, unsigned int width, unsigned int height, unsigned int depth, 
                         int num_iterations) {
    
    size_t size = width * height * depth;
    
    // Allocate result arrays
    int* d_index_pba;
    int* d_index_edt;
    int* d_index_brute;
    float* d_distance_pba;
    float* d_distance_edt;
    float* d_distance_brute;
    
    cudaMalloc(&d_index_pba, size * 3 * sizeof(int));
    cudaMalloc(&d_index_edt, size * 3 * sizeof(int));
    cudaMalloc(&d_index_brute, size * 3 * sizeof(int));
    cudaMalloc(&d_distance_pba, size * sizeof(float));
    cudaMalloc(&d_distance_edt, size * sizeof(float));
    cudaMalloc(&d_distance_brute, size * sizeof(float));
    
    // Warm up
    pba_edt_3d(d_boundary, d_index_pba, d_distance_pba, width, height, depth);
    
    // Benchmark PBA
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("Benchmarking PBA EDT...\n");
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        pba_edt_3d(d_boundary, d_index_pba, d_distance_pba, width, height, depth);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float pba_time;
    cudaEventElapsedTime(&pba_time, start, stop);
    
    // Benchmark proper CUDA EDT
    printf("Benchmarking Proper CUDA EDT...\n");
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        edt_3d(d_boundary, d_index_edt, d_distance_edt, width, height, depth);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float edt_time;
    cudaEventElapsedTime(&edt_time, start, stop);
    
    // Benchmark brute-force EDT (only if volume is very small)
    float brute_time = 0.0f;
    if (size <= 32*32*32) { // Only for very small volumes to avoid timeout
        printf("Benchmarking Brute-force EDT...\n");
        cudaEventRecord(start);
        for (int i = 0; i < std::min(num_iterations, 2); i++) { // Fewer iterations for brute force
            brute_force_edt_3d(d_boundary, d_index_brute, d_distance_brute, width, height, depth);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&brute_time, start, stop);
        brute_time /= std::min(num_iterations, 2);
    }
    
    printf("=== Performance Comparison ===\n");
    printf("PBA EDT Time:         %.3f ms (average over %d iterations)\n", 
           pba_time / num_iterations, num_iterations);
    printf("Proper CUDA EDT Time: %.3f ms (average over %d iterations)\n", 
           edt_time / num_iterations, num_iterations);
    printf("PBA Speedup:          %.1fx faster than CUDA EDT\n", 
           edt_time / pba_time);
    
    if (brute_time > 0.0f) {
        printf("Brute-force EDT Time: %.3f ms (average over %d iterations)\n", 
               brute_time, std::min(num_iterations, 2));
        printf("PBA vs Brute-force:   %.1fx faster\n", brute_time / (pba_time / num_iterations));
        printf("EDT vs Brute-force:   %.1fx faster\n", brute_time / (edt_time / num_iterations));
    }
    
    // Clean up
    cudaFree(d_index_pba);
    cudaFree(d_index_edt);
    cudaFree(d_index_brute);
    cudaFree(d_distance_pba);
    cudaFree(d_distance_edt);
    cudaFree(d_distance_brute);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
