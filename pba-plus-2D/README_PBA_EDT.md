# PBA 2D Euclidean Distance Transform API

This directory contains a 2D Euclidean Distance Transform (EDT) API built on top of the Parallel Banding Algorithm (PBA). It provides a convenient interface for computing distance transforms and Voronoi diagrams in 2D.

## Files Overview

- `pba2d_edt_api.hpp` - Header file with EDT API function declarations
- `pba2d_edt_api.cu` - Implementation of the EDT API using PBA
- `pba2d_wrapper_clean.hpp` - C++ wrapper class for easier PBA usage
- `test_pba_edt.cpp` - Test program demonstrating the EDT API
- `wrapper_example.cpp` - Example program showing the wrapper class usage
- `Makefile` - Build configuration for all components

## API Functions

### Core EDT Functions

```cpp
// Basic PBA-based EDT with arbitrary dimensions
void pba_edt_2d(char* d_boundary, int* index, float* distance, 
                unsigned int width, unsigned int height);

// Optimized version for square power-of-2 dimensions
void pba_edt_2d_optimized(char* d_boundary, int* index, float* distance, 
                          unsigned int size);
```

### Comparison Functions

```cpp
// Standard EDT implementation for comparison
void edt_2d(char* d_boundary, int* index, float* distance, 
            unsigned int width, unsigned int height);

// Brute force EDT for verification
void brute_force_edt_2d(char* d_boundary, int* index, float* distance,
                        unsigned int width, unsigned int height);

// Performance benchmark
void benchmark_pba_vs_edt_2d(char* d_boundary, unsigned int width, unsigned int height, 
                             int num_iterations = 10);
```

### Input/Output Format

- **Input**: `char* d_boundary` - Binary mask where 1 = boundary pixel, 0 = empty
- **Output**: 
  - `int* index` - Array of nearest boundary pixel coordinates [height*width*2]
  - `float* distance` - Array of distances to nearest boundary [height*width]

## C++ Wrapper Class

The `PBA::PBA2D` class provides a more convenient interface:

```cpp
#include "pba2d_wrapper_clean.hpp"
using namespace PBA;

PBA2D pba;
pba.initialize(128);  // Power of 2 size

// Add seeds
pba.addSeed(32, 32);
pba.addSeed(96, 96);

// Or add from mask
char mask[128*128];
// ... fill mask ...
pba.addSeedsFromMask(mask);

// Compute
pba.computeVoronoi();

// Query results
QueryResult2D result = pba.query(50, 50);
printf("Distance: %.2f, Nearest: (%d, %d)\n", 
       result.distance, result.nearest_x, result.nearest_y);
```

## Building

Requirements:
- NVIDIA CUDA Toolkit
- GPU with compute capability 3.0+

```bash
# Build all targets
make

# Build with debug info
make debug

# Run tests
make test

# Run wrapper example
make example

# Clean build files
make clean
```

## Usage Examples

### Example 1: Basic EDT Computation

```cpp
#include "pba2d_edt_api.hpp"

int width = 256, height = 256;
size_t size = width * height;

// Allocate device memory
char* d_boundary;
int* d_index;
float* d_distance;
cudaMalloc(&d_boundary, size * sizeof(char));
cudaMalloc(&d_index, size * 2 * sizeof(int));
cudaMalloc(&d_distance, size * sizeof(float));

// Set up boundary mask (1 = boundary, 0 = empty)
// ... fill d_boundary ...

// Compute EDT
pba_edt_2d(d_boundary, d_index, d_distance, width, height);

// Use results...
```

### Example 2: Using the Wrapper Class

```cpp
#include "pba2d_wrapper_clean.hpp"
using namespace PBA;

PBA2D pba;
pba.initialize(256);

// Add corner seeds
pba.addSeed(0, 0);
pba.addSeed(255, 0);
pba.addSeed(0, 255);
pba.addSeed(255, 255);

// Compute
pba.computeVoronoi();

// Get complete results
VoronoiResult2D result = pba.getResult();
for (int i = 0; i < result.total_elements; i++) {
    printf("Distance[%d] = %.2f\n", i, result.distances[i]);
}

freeVoronoiResult(result);
```

## Performance Characteristics

- **Speed**: Generally faster than brute-force EDT, especially for large images
- **Memory**: Requires O(n²) memory for n×n images
- **Accuracy**: Exact Euclidean distances (floating-point precision)
- **Limitations**: 
  - Optimized version requires square power-of-2 dimensions
  - General version handles arbitrary dimensions but may be slower

## Algorithm Details

The implementation uses the Parallel Banding Algorithm (PBA) for computing 2D Voronoi diagrams, then derives distance transforms from the Voronoi results. Key features:

1. **GPU Acceleration**: All computations run on CUDA-capable GPUs
2. **Memory Efficient**: Uses PBA's optimized memory layout
3. **Flexible Input**: Supports arbitrary rectangular dimensions
4. **Standard Output**: Compatible with common EDT result formats

## Testing

The test suite includes:

1. **Correctness Tests**: Compare PBA-EDT with brute-force EDT
2. **Performance Tests**: Benchmark against standard implementations
3. **Edge Cases**: Various image sizes and boundary patterns
4. **Visual Verification**: Sample outputs for manual inspection

Run tests with:
```bash
make test
./pba_edt_test_2d
```

## File Output

Test programs generate:
- `pba_distances_2d.dat` - Distance values (binary float array)
- `pba_indices_2d.dat` - Index values (binary int array)
- `boundary_mask_2d.dat` - Input boundary mask (binary char array)

## Troubleshooting

**Common Issues:**

1. **"CUDA out of memory"**: Reduce image size or use optimized version
2. **"Invalid texture size"**: Ensure size is power of 2 for optimized version
3. **Incorrect results**: Check that boundary mask uses values 0 and 1 only
4. **Poor performance**: Use power-of-2 dimensions when possible

**Debug Tips:**

```bash
# Build with debug info
make debug

# Check CUDA installation
make check

# Verify build variables
make print-vars
```

## Related Files

This 2D implementation follows the same patterns as the 3D version in `../pba-plus-3D/`. The main differences:

- 2D uses `short` pairs instead of `int` encoding for coordinates
- Simpler memory layout (width × height vs width × height × depth)
- Different PBA function calls (`pba2D*` vs `pba3D*`)
- 2-component indices instead of 3-component

## References

- Original PBA paper: Cao, T.T., Tang, K., Mohamed, A., Tan, T.S. (2010)
- Project homepage: http://www.comp.nus.edu.sg/~tants/pba.html
