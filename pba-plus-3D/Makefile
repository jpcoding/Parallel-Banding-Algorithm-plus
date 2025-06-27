# Makefile for PBA3D project

# Compiler settings
NVCC = nvcc
ARCH = sm_80
CFLAGS = -arch=$(ARCH)

# Source files
MAIN_SRC = main.cpp
PBA_SRC = ./pba/pba3DHost.cu
WRAPPER_EXAMPLE_SRC = wrapper_example.cpp
PBA_EDT_TEST_SRC = test_pba_edt.cpp
PBA_EDT_API_SRC = pba3d_edt_api.cu

# Targets
all: test wrapper_example pba_edt_test

test: $(MAIN_SRC) $(PBA_SRC)
	$(NVCC) $(CFLAGS) -o test $(MAIN_SRC) $(PBA_SRC)

wrapper_example: $(WRAPPER_EXAMPLE_SRC) $(PBA_SRC)
	$(NVCC) $(CFLAGS) -o wrapper_example $(WRAPPER_EXAMPLE_SRC) $(PBA_SRC)

pba_edt_test: $(PBA_EDT_TEST_SRC) $(PBA_SRC) $(PBA_EDT_API_SRC)
	$(NVCC) $(CFLAGS) -o pba_edt_test $(PBA_EDT_TEST_SRC) $(PBA_SRC) $(PBA_EDT_API_SRC)

clean:
	rm -f test wrapper_example pba_edt_test *.dat

.PHONY: all clean
