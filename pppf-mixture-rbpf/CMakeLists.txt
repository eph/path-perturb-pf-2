cmake_minimum_required(VERSION 3.16)

project(pppf_mixture_rbpf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)

add_executable(burnside_validate
  src/burnside_validate.cpp
)
target_include_directories(burnside_validate PRIVATE include)
target_link_libraries(burnside_validate PRIVATE Eigen3::Eigen)

add_executable(nk_compare
  src/nk_compare.cpp
)
target_include_directories(nk_compare PRIVATE include)
target_link_libraries(nk_compare PRIVATE Eigen3::Eigen)
