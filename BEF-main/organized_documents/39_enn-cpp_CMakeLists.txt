cmake_minimum_required(VERSION 3.15)
project(ENN_CPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

# Core ENN library
add_library(enn STATIC
    src/cell.cpp
    src/collapse.cpp
    src/regularizers.cpp
    src/optim.cpp
    src/data.cpp
)

target_include_directories(enn PUBLIC include)
target_link_libraries(enn PUBLIC Eigen3::Eigen)

# Speed flags — tune for your CPU
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(enn PRIVATE -O3 -march=native -ffast-math -fno-math-errno)
endif()

# Applications
add_executable(committor_train apps/committor_train.cpp)
target_link_libraries(committor_train PRIVATE enn)

add_executable(seq_demo apps/seq_demo.cpp)
target_link_libraries(seq_demo PRIVATE enn)

# Tests
enable_testing()

add_executable(test_softmax tests/test_softmax.cpp)
target_link_libraries(test_softmax PRIVATE enn)
add_test(NAME SoftmaxTest COMMAND test_softmax)

add_executable(test_psd tests/test_psd.cpp)
target_link_libraries(test_psd PRIVATE enn)
add_test(NAME PSDTest COMMAND test_psd)

add_executable(test_gradcheck tests/test_gradcheck.cpp)
target_link_libraries(test_gradcheck PRIVATE enn)
add_test(NAME GradCheckTest COMMAND test_gradcheck)