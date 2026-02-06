cmake_minimum_required(VERSION 3.16)
project(geomzk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(geomzk STATIC
    src/hash.cpp
    src/json.cpp
    src/proof_loader.cpp
    src/verify.cpp
)

target_include_directories(geomzk PUBLIC include)

add_executable(geomzk_cli src/geomzk_cli.cpp)
target_link_libraries(geomzk_cli PRIVATE geomzk)
