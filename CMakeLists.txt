cmake_minimum_required(VERSION 3.14)

project(Tucil2_13524127_13524132)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Mengatur agar executable masuk ke folder bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(Tucil2_13524127_13524132
    src/main.cpp
    src/octree.cpp
    src/obj_io.cpp
    src/voxelizer.cpp
    src/viewer.cpp
)
