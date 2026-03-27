cmake_minimum_required(VERSION 3.14)

project(Tucil2_13524127_13524132)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Mengatur agar executable masuk ke folder bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

include(FetchContent)
set(BUILD_EXAMPLES OFF CACHE BOOL "Build raylib examples" FORCE)
set(BUILD_GAMES OFF CACHE BOOL "Build raylib games" FORCE)
FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG 5.5
)
FetchContent_MakeAvailable(raylib)

add_executable(Tucil2_13524127_13524132
    src/main.cpp
    src/octree.cpp
    src/obj_io.cpp
    src/voxelizer.cpp
    src/viewer.cpp
)

target_link_libraries(Tucil2_13524127_13524132 PRIVATE raylib)
