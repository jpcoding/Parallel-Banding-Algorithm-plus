# Makefile for PBA 2D EDT API

# CUDA compiler
NVCC = nvcc

# CUDA architecture (adjust based on your GPU)
CUDA_ARCH = -arch=sm_80

# Compiler flags
NVCC_FLAGS = -O3 -std=c++11 $(CUDA_ARCH)
DEBUG_FLAGS = -g -G

# Directories
PBA_DIR = pba
SRC_DIR = .
BUILD_DIR = build

# Source files
PBA_SOURCES = $(PBA_DIR)/pba2DHost.cu
API_SOURCES = pba2d_edt_api.cu
TEST_SOURCES = test_pba_edt.cpp
WRAPPER_EXAMPLE_SOURCES = wrapper_example.cpp

# Object files
PBA_OBJECTS = $(BUILD_DIR)/pba2DHost.o
API_OBJECTS = $(BUILD_DIR)/pba2d_edt_api.o

# Executables
TEST_EXEC = pba_edt_test_2d
WRAPPER_EXEC = wrapper_example_2d

# Default target
all: $(TEST_EXEC) $(WRAPPER_EXEC)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile PBA objects
$(BUILD_DIR)/pba2DHost.o: $(PBA_DIR)/pba2DHost.cu $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile API objects
$(BUILD_DIR)/pba2d_edt_api.o: pba2d_edt_api.cu $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Build test executable
$(TEST_EXEC): $(TEST_SOURCES) $(PBA_OBJECTS) $(API_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

# Build wrapper example
$(WRAPPER_EXEC): $(WRAPPER_EXAMPLE_SOURCES) $(PBA_OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

# Debug versions
debug: NVCC_FLAGS += $(DEBUG_FLAGS)
debug: $(TEST_EXEC) $(WRAPPER_EXEC)

# Clean targets
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(TEST_EXEC) $(WRAPPER_EXEC)
	rm -f *.dat

# Clean all (including data files)
cleanall: clean
	rm -f *.dat *.log

# Install (copy to system directories - optional)
install: $(TEST_EXEC) $(WRAPPER_EXEC)
	cp $(TEST_EXEC) /usr/local/bin/
	cp $(WRAPPER_EXEC) /usr/local/bin/

# Uninstall
uninstall:
	rm -f /usr/local/bin/$(TEST_EXEC)
	rm -f /usr/local/bin/$(WRAPPER_EXEC)

# Run tests
test: $(TEST_EXEC)
	./$(TEST_EXEC)

# Run wrapper example
example: $(WRAPPER_EXEC)
	./$(WRAPPER_EXEC)

# Check CUDA setup
check:
	@echo "Checking CUDA installation..."
	@which $(NVCC) || echo "NVCC not found"
	@$(NVCC) --version || echo "NVCC version check failed"

# Print variables for debugging
print-vars:
	@echo "NVCC = $(NVCC)"
	@echo "NVCC_FLAGS = $(NVCC_FLAGS)"
	@echo "CUDA_ARCH = $(CUDA_ARCH)"
	@echo "PBA_SOURCES = $(PBA_SOURCES)"
	@echo "API_SOURCES = $(API_SOURCES)"

# Help
help:
	@echo "Available targets:"
	@echo "  all        - Build all executables"
	@echo "  debug      - Build with debug flags"
	@echo "  test       - Run the EDT test"
	@echo "  example    - Run the wrapper example"
	@echo "  clean      - Remove build files and executables"
	@echo "  cleanall   - Remove all generated files"
	@echo "  check      - Check CUDA installation"
	@echo "  help       - Show this help message"

.PHONY: all debug clean cleanall install uninstall test example check print-vars help
