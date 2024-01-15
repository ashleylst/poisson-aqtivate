cmake_minimum_required(VERSION 3.26)
project(poisson C)

set(CMAKE_C_STANDARD 11)

add_executable(test check_poisson.c)
