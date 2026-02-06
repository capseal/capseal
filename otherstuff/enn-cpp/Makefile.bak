# Simple Makefile for ENN C++ (without CMake)

CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -ffast-math -Wall -Wextra -Iinclude -fopenmp
LDFLAGS = -fopenmp 

# Try to find Eigen
EIGEN_PATHS = third_party/eigen /usr/include/eigen3 /usr/local/include/eigen3 /opt/homebrew/include/eigen3 \
              /usr/include /usr/local/include /opt/homebrew/include
EIGEN_FOUND = $(shell for path in $(EIGEN_PATHS); do if [ -d "$$path/Eigen" ]; then echo "$$path"; break; fi; done)

ifneq ($(EIGEN_FOUND),)
    CXXFLAGS += -I$(EIGEN_FOUND)
    $(info Found Eigen at: $(EIGEN_FOUND))
else
    $(info Eigen not found. Downloading...)
    $(shell mkdir -p third_party && cd third_party && wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && tar -xzf eigen-3.4.0.tar.gz && ln -sf eigen-3.4.0 eigen)
    CXXFLAGS += -Ithird_party/eigen
    $(info Downloaded Eigen to third_party/eigen)
endif

# Source files
SRCDIR = src
SOURCES = $(SRCDIR)/cell.cpp $(SRCDIR)/collapse.cpp $(SRCDIR)/regularizers.cpp $(SRCDIR)/data.cpp $(SRCDIR)/trainer.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# Applications
APPS = apps/committor_train apps/seq_demo apps/seq_demo_bptt apps/seq_debug apps/bicep_to_enn

# Tests  
TESTS = tests/test_softmax tests/test_psd tests/test_gradcheck tests/test_bptt_gradcheck

# Default target
all: $(APPS) $(TESTS)

# Library object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Applications
apps/committor_train: apps/committor_train.cpp $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

apps/seq_demo: apps/seq_demo.cpp $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

apps/seq_demo_bptt: apps/seq_demo_bptt.cpp $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

apps/seq_debug: apps/seq_debug.cpp $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

apps/bicep_to_enn: apps/bicep_to_enn.cpp $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Tests
tests/test_softmax: tests/test_softmax.cpp $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

tests/test_psd: tests/test_psd.cpp $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

tests/test_gradcheck: tests/test_gradcheck.cpp $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

tests/test_bptt_gradcheck: tests/test_bptt_gradcheck.cpp $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Run tests
test: $(TESTS)
	@echo "Running tests..."
	@for test in $(TESTS); do \
		echo "Running $$test..."; \
		./$$test || exit 1; \
	done
	@echo "All tests passed!"

# Demo target
demo: apps/committor_train apps/seq_demo_bptt
	@echo "Running committor training demo..."
	./apps/committor_train
	@echo ""
	@echo "Running BPTT sequence demo..."  
	./apps/seq_demo_bptt

# Clean
clean:
	rm -f $(OBJECTS) $(APPS) $(TESTS)

# Check if we can compile a simple Eigen program
check:
	@echo "Testing Eigen compilation..."
	@echo '#include <Eigen/Dense>' > test_eigen.cpp
	@echo '#include <iostream>' >> test_eigen.cpp
	@echo 'int main() { Eigen::MatrixXd m(2,2); m << 1,2,3,4; std::cout << "Eigen works!" << std::endl; return 0; }' >> test_eigen.cpp
	@$(CXX) $(CXXFLAGS) test_eigen.cpp -o test_eigen $(LDFLAGS) && ./test_eigen && rm test_eigen test_eigen.cpp || echo "Eigen test failed - please install Eigen3"

.PHONY: all test demo clean check