cmake_minimum_required(VERSION 3.15)
project(PAAD VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(netCDF REQUIRED)
find_package(OpenMP REQUIRED)

set(SOURCES
    src/adjoint.cpp
    src/atmosphere.cpp
    src/configuration.cpp
    src/forward.cpp
    src/geometry.cpp
    src/json.cpp
    src/math.cpp
    src/mie.cpp
    src/physics.cpp
    src/radiative_transfer.cpp
    src/rayleigh.cpp
    src/solver.cpp
    src/inversion.cpp
    src/string.cpp
    src/units.cpp
    src/vira.cpp
    src/io.cpp
    src/tools.cpp
	src/logger.cpp
	src/sfi.cpp
)

add_library(paad_lib ${SOURCES})

target_include_directories(paad_lib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

target_link_libraries(paad_lib PUBLIC Eigen3::Eigen netCDF::netcdf netcdf_c++4 OpenMP::OpenMP_CXX)

add_executable(paad_run paad_run.cpp)
target_link_libraries(paad_run PRIVATE paad_lib)

add_executable(paad_make_source paad_make_source.cpp)
target_link_libraries(paad_make_source PRIVATE paad_lib)
