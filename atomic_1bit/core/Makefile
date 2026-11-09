CXX = g++
CXXFLAGS = -O3 -fPIC -shared
TARGET = libatomic.so

# Default backend
BACKEND ?= CPU

ifeq ($(BACKEND), CPU)
	SRCS = backends/cpu_kernel.cpp
	# CXXFLAGS += -fopenmp  # specific to GCC/libomp
	# Optional: enable the AVX2 path only on x86_64 hosts that actually
	# expose it. Without __AVX2__ the kernel builds the scalar fallback,
	# so this stays safe on every platform (ARM keeps NEON via __ARM_NEON).
	ifeq ($(shell uname -m),x86_64)
		ifeq ($(shell uname -s),Linux)
			HAVE_AVX2 := $(shell grep -qw avx2 /proc/cpuinfo 2>/dev/null && echo yes)
		endif
		ifeq ($(shell uname -s),Darwin)
			HAVE_AVX2 := $(shell sysctl -n machdep.cpu.leaf7_features 2>/dev/null | grep -qw AVX2 && echo yes)
		endif
		ifeq ($(HAVE_AVX2),yes)
			CXXFLAGS += -mavx2
		endif
	endif
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
	rm -f $(TARGET) backends/*.air backends/*.metallib default.metallib
