cmake_minimum_required(VERSION 3.26)
project(poisson C)

set(CMAKE_C_STANDARD 11)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    MESSAGE("openmp found")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

include(CheckLibraryExists)

CHECK_LIBRARY_EXISTS(m sin "" HAVE_LIB_M)

if (HAVE_LIB_M)
    set(EXTRA_LIBS ${EXTRA_LIBS} m)
endif (HAVE_LIB_M)

add_executable(poisson main.c
        solver.c
        mpi_solver.c
        array_helpers.c
        array_helpers.h
        log.c
        log.h
        test.c
        test.h)

target_link_libraries(poisson ${MPI_C_LIBRARIES})
target_link_libraries(poisson ${EXTRA_LIBS})