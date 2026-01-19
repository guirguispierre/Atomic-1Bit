CXX = g++
CXXFLAGS = -O3 -fPIC -shared
TARGET = libatomic.so

# Default backend
BACKEND ?= CPU

ifeq ($(BACKEND), CPU)
	SRCS = backends/cpu_kernel.cpp
	# CXXFLAGS += -fopenmp  # specific to GCC/libomp
endif

ifeq ($(BACKEND), METAL)
	SRCS = backends/metal_kernel.mm
	CXXFLAGS += -framework Metal -framework Foundation -lobjc
endif

ifeq ($(BACKEND), CUDA)
	CXX = nvcc
	SRCS = backends/cuda_kernel.cu
	CXXFLAGS = -O3 --compiler-options '-fPIC' -shared
endif

all: $(TARGET)
	@echo "Built for backend: $(BACKEND)"

$(TARGET): $(SRCS)
ifeq ($(BACKEND), CUDA)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)
else
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)
endif
# Compile Metal Shaders if needed
ifeq ($(BACKEND), METAL)
	xcrun -sdk macosx metal -c backends/kernels.metal -o backends/kernels.air
	xcrun -sdk macosx metallib backends/kernels.air -o default.metallib
endif

clean:
	rm -f $(TARGET) backends/*.air backends/*.metallib
