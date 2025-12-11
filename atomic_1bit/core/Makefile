CXX = g++
CXXFLAGS = -O3 -fPIC -shared
TARGET = libatomic.so
SRCS = kernel.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)
