cmake_minimum_required(VERSION 2.8)
find_package(OpenMP REQUIRED)
set(CMAKE_BUILD_TYPE "Release")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

add_executable(p1 p1.c)
target_link_libraries(p1 m)
