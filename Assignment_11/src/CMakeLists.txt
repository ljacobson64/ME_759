cmake_minimum_required(VERSION 2.8)
find_package(MPI REQUIRED)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_C_COMPILE_FLAGS ${CMAKE_C_COMPILE_FLAGS} ${MPI_C_COMPILE_FLAGS})
set(CMAKE_C_LINK_FLAGS ${CMAKE_C_LINK_FLAGS} ${MPI_LINK_FLAGS})

include_directories(${MPI_INCLUDE_PATH})

add_executable(p1 p1.c)
target_link_libraries(p1 ${MPI_LIBRARIES})
# target_link_libraries(p1 m)
